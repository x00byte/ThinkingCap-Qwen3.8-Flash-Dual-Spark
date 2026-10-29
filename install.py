"""Install and run ThinkingCap on two Sparks from the head host."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import fcntl
import io
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'cluster.local.json'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes']
TRANSFER_ROOTS = ['install.py', 'install.sh', 'serve.py', 'models.py', 'models.lock.json',
                  'config.example.json', 'README.md', 'THIRD_PARTY.md', '.gitignore',
                  'runtime', 'profiles', 'tests', 'docs', 'assets', 'benchmarks']


def execute(args, *, capture=True, input=None, timeout=None):
    result = subprocess.run(args, input=input, stdout=subprocess.PIPE if capture else None,
                            check=False, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{Path(args[0]).name} failed (exit {result.returncode}); see the output above')
    return result.stdout.decode().strip() if capture else None


def ssh(worker, args, **kwargs):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', worker):
        raise ValueError('Use a worker SSH alias or user@hostname, without options')
    return execute(['ssh', *SSH_OPTIONS, worker, shlex.join(args)], **kwargs)


def discover_fabric(sysfs=Path('/sys/class/infiniband'), addresses=None):
    if addresses is None:
        addresses = json.loads(execute(['ip', '-j', '-4', 'addr', 'show']))
    networks = {(row['ifname'], addr['local']): addr['prefixlen']
                for row in addresses for addr in row.get('addr_info', [])
                if addr['family'] == 'inet'}
    result = []
    for hca in sorted(sysfs.glob('*')):
        port = hca / 'ports/1'
        try:
            if 'ACTIVE' not in (port / 'state').read_text():
                continue
            for gid in sorted((port / 'gids').glob('*')):
                if (port / 'gid_attrs/types' / gid.name).read_text().strip() != 'RoCE v2':
                    continue
                addr = ipaddress.IPv6Address(gid.read_text().strip()).ipv4_mapped
                interface = (port / 'gid_attrs/ndevs' / gid.name).read_text().strip()
                if addr is not None and (interface, str(addr)) in networks:
                    result.append({'address': str(addr), 'interface': interface,
                                   'hca': hca.name, 'gid_index': gid.name,
                                   'prefixlen': networks[interface, str(addr)]})
        except OSError:
            continue
    return result


def probe():
    if sys.version_info < (3, 10) or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Each Spark needs ARM64 Linux and Python 3.10 or newer')
    for program in ('docker', 'nvidia-smi', 'ip', 'ping'):
        if shutil.which(program) is None:
            raise RuntimeError(f'Missing {program}; configure the Spark host first')
    info = json.loads(execute(['docker', 'info', '--format', '{{json .}}']))
    if 'nvidia' not in info.get('Runtimes', {}):
        raise RuntimeError('Configure NVIDIA Container Toolkit for Docker first')
    gpu = execute(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    if 'GB10' not in gpu:
        raise RuntimeError('This profile requires an NVIDIA GB10 GPU on each host')
    fabric = discover_fabric()
    if not fabric:
        raise RuntimeError('No active IPv4 RoCE v2 GID found; configure the fabric first')
    return {'home': str(Path.home()), 'fabric': fabric, 'gpu': gpu}


def select_fabric(head, worker, head_address=None, worker_address=None):
    pairs = []
    for a in head:
        for b in worker:
            an = ipaddress.ip_interface(f"{a['address']}/{a['prefixlen']}").network
            bn = ipaddress.ip_interface(f"{b['address']}/{b['prefixlen']}").network
            if a['address'] != b['address'] and ipaddress.ip_address(b['address']) in an and ipaddress.ip_address(a['address']) in bn:
                pairs.append((a, b))
    pairs.sort(key=lambda pair: (int(ipaddress.ip_address(pair[0]['address'])),
                                  int(pair[0]['gid_index']), pair[0]['hca']))
    eligible = [(a, b) for a, b in pairs
                if (not head_address or a['address'] == head_address)
                and (not worker_address or b['address'] == worker_address)]
    if not eligible:
        raise RuntimeError('No shared RoCE subnet found. Check the fabric or supply --head-address and --worker-address.')
    chosen = [eligible[0]]
    # Include an additional rail only when both sides have a distinct matching HCA
    # and its GID index agrees with the primary rail on its own host.
    for a, b in pairs:
        if (a['gid_index'] == chosen[0][0]['gid_index'] and b['gid_index'] == chosen[0][1]['gid_index']
                and a['hca'] not in [x['hca'] for x, _ in chosen]
                and b['hca'] not in [y['hca'] for _, y in chosen]):
            chosen.append((a, b))
    return chosen


def recipe_archive():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as tar:
        for name in TRANSFER_ROOTS:
            root = ROOT / name
            if root.is_symlink():
                raise ValueError(f'Refusing a symlink in recipe roots: {root}')
            paths = sorted(root.rglob('*')) if root.is_dir() else [root]
            for path in paths:
                if any(part in ('__pycache__', '.cache', '.venv', '.git') for part in path.relative_to(ROOT).parts) or path.name.startswith('.env') or path.suffix in ('.pyc', '.pyo', '.log', '.safetensors', '.bin', '.pt', '.pem', '.key'):
                    continue
                if path.is_symlink():
                    raise ValueError(f'Refusing a symlink in recipe files: {path}')
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    return output.getvalue()


# Extract only regular files inside the dedicated recipe directory. This also
# rejects an existing parent symlink that would redirect writes outside it.
EXTRACT = '''import pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1])
if root.is_symlink():raise ValueError("Recipe directory must not be a symlink")
root.mkdir(parents=True,exist_ok=True);root=root.resolve()
marker=root/".thinkingcap-installer"
if marker.is_symlink():raise ValueError("Recipe marker must not be a symlink")
if any(root.iterdir()) and (not marker.is_file() or marker.read_text()!="x00byte.thinkingcap-qwen38"):
 raise ValueError("Worker recipe directory already contains files not managed by this installer")
marker.write_text("x00byte.thinkingcap-qwen38")
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|") as archive:
 for member in archive:
  rel=pathlib.PurePosixPath(member.name)
  if rel.is_absolute() or ".." in rel.parts or not member.isfile():raise ValueError("Unsafe recipe archive")
  path=root.joinpath(*rel.parts)
  if not path.resolve().is_relative_to(root):raise ValueError("Recipe symlink escapes destination")
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open("wb") as output:output.write(archive.extractfile(member).read())
  path.chmod(member.mode & 0o777)
'''


def config_for(base, fabric, peer, data_dir):
    cfg = dict(base)
    if data_dir:
        data = Path(data_dir)
        if not data.is_absolute():
            raise ValueError('Data directories must be absolute paths')
        cfg.update(model_dir=str(data / 'models/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4'),
                   draft_dir=str(data / 'models/incoai/Qwen3.8-Flash-Next-DFlash2'),
                   cache_dir=str(data / 'cache'))
    cfg.update(head_addr=peer[0], worker_addr=peer[1], fabric_interface=fabric[0]['interface'],
               fabric_hca=','.join(row['hca'] for row in fabric), gid_index=fabric[0]['gid_index'])
    import serve
    return serve.validate(cfg)


def save_config(proposed):
    path = ROOT / 'config.local.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != proposed:
            raise RuntimeError(f'{path} already contains different settings; preserve or move it before installing')
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(proposed, indent=2) + '\n')
    temporary.replace(path)


def image_ready(cfg):
    import serve
    result = subprocess.run(['docker', 'image', 'inspect', cfg['image']], capture_output=True, text=True)
    if result.returncode:
        return False
    image = json.loads(result.stdout)[0]
    return image['Architecture'] == 'arm64' and (image['Config'].get('Labels') or {}).get('org.thinkingcap.overlay') == serve.overlay_fingerprint()


def running_service(cfg, rank):
    import serve
    info = serve.container(cfg)
    if info is None:
        return None
    labels = info.get('Config', {}).get('Labels') or {}
    if labels.get('org.thinkingcap.owner') != serve.OWNER:
        raise RuntimeError('The chosen container name belongs to another service')
    if (not info['State']['Running'] or labels.get('org.thinkingcap.config') != serve.config_fingerprint(cfg, rank)
            or not image_ready(cfg) or info['Image'] != json.loads(serve.run(['docker', 'image', 'inspect', cfg['image']]))[0]['Id']):
        raise RuntimeError('A previous project container has different settings or has exited. Inspect its logs, then use serve.py stop on this host before retrying.')
    return info


def ensure_idle(cfg, rank):
    if running_service(cfg, rank):
        return
    if execute(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        raise RuntimeError('GPU is occupied. Drain and stop the current service with its own launcher first.')


def free_space(cfg, manifests):
    totals = {}
    for model, key in zip(manifests, ('model_dir', 'draft_dir')):
        directory = Path(cfg[key]); directory.mkdir(parents=True, exist_ok=True)
        missing = sum(row['bytes'] for row in model['files']
                      if not (directory / row['path']).is_file()
                      or (directory / row['path']).stat().st_size != row['bytes'])
        dev = directory.stat().st_dev
        totals.setdefault(dev, [directory, 0])[1] += missing
    for directory, needed in totals.values():
        if shutil.disk_usage(directory).free < needed + 20 * 1024**3:
            raise RuntimeError(f'Not enough free space at {directory}: need {needed / 1e9:.1f} GB of missing weights plus 20 GiB working space')


def prepare_node(cfg, rank, prepare_only=False):
    import models
    import serve
    if running_service(cfg, rank):
        print('Matching service is already running; reusing it.', flush=True)
        return
    if not prepare_only:
        ensure_idle(cfg, rank)
    serve.validate(cfg)
    save_config(cfg)
    serve.load(ROOT / 'config.local.json')
    manifests = json.loads((ROOT / 'models.lock.json').read_text())
    free_space(cfg, manifests)
    execute([sys.executable, str(ROOT / 'runtime/verify_overlay.py')], capture=False)
    if not image_ready(cfg):
        execute(['docker', 'build', '-t', cfg['image'], str(ROOT / 'runtime')], capture=False)
    # Hub tooling comes from the pinned runtime. No host pip installation or GPU
    # access is needed. A supplied HF_TOKEN is passed by environment name only.
    download = ('from huggingface_hub import snapshot_download;import json,sys;'
                'snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],'
                'local_dir="/download",allow_patterns=json.loads(sys.argv[3]),force_download=bool(int(sys.argv[4])))')
    for model, key in zip(manifests, ('model_dir', 'draft_dir')):
        directory = Path(cfg[key])
        invalid = []
        for row in model['files']:
            try:
                models.verify(directory, {'files': [row]})
            except ValueError:
                invalid.append(row)
        if not invalid:
            print(f"Verified existing {model['repo']}", flush=True)
            continue
        needed = sum(row['bytes'] for row in invalid)
        if shutil.disk_usage(directory).free < needed + 5 * 1024**3:
            raise RuntimeError(f'Not enough room to download or repair {model["repo"]}')
        force = any((directory / row['path']).exists() for row in invalid)
        print(f"Downloading {model['repo']} at {model['revision']}", flush=True)
        args = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
                '-e', 'HOME=/tmp', '-e', 'HF_HOME=/download/.cache/huggingface',
                '-v', str(directory) + ':/download', '--entrypoint', 'python3']
        if os.environ.get('HF_TOKEN'):
            args += ['-e', 'HF_TOKEN']
        args += [cfg['image'], '-c', download, model['repo'], model['revision'],
                 json.dumps([row['path'] for row in invalid]), str(int(force))]
        execute(args, capture=False)
        models.verify(directory, model)
        print(f"Verified {model['repo']}", flush=True)
    execute(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3',
             '-v', str(ROOT) + ':/recipe:ro', cfg['image'], '/recipe/runtime/check_contract.py'], capture=False)


def start_node(cfg, rank):
    import serve
    info = running_service(cfg, rank)
    if info:
        return {'created': False, 'id': info['Id']}
    with redirect_stdout(sys.stderr):
        serve.start(cfg, rank)
    info = serve.container(cfg)
    if info is None:
        raise RuntimeError('New container disappeared after creation')
    return {'created': True, 'id': info['Id']}


def remove_created(cfg, container_id):
    import serve
    info = serve.container(cfg)
    if (info is not None and info['Id'] == container_id
            and (info.get('Config', {}).get('Labels') or {}).get('org.thinkingcap.owner') == serve.OWNER):
        try:
            logs = subprocess.run(['docker', 'logs', '--tail', '100', container_id], capture_output=True, text=True)
            destination = Path(cfg['cache_dir']) / 'install-startup.log'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(logs.stdout + logs.stderr)
            print(f'Startup log saved at {destination}', file=sys.stderr)
        except OSError as exc:
            print(f'Could not save startup log: {exc}', file=sys.stderr)
        serve.run(['docker', 'rm', '-f', container_id])


def wait_ready(cfg, worker, recipe, timeout):
    import serve
    deadline = time.monotonic() + timeout
    next_worker_check = 0
    while time.monotonic() < deadline:
        info = serve.container(cfg)
        if not info or not info['State']['Running']:
            raise RuntimeError('Head container exited during startup')
        if time.monotonic() >= next_worker_check:
            state = json.loads(ssh(worker, ['python3', str(Path(recipe) / 'install.py'), '--node-status']))
            if not state['running']:
                raise RuntimeError('Worker container exited during startup')
            next_worker_check = time.monotonic() + 15
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/health", timeout=3) as response:
                if response.status == 200:
                    with urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=3) as response:
                        entries = json.load(response)['data']
                    if any(row['id'] == cfg['served_model_name'] and row.get('max_model_len') == 262144 for row in entries):
                        return
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError('API did not become ready within the startup timeout')


def install(args):
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    worker = args.worker or previous.get('worker')
    if not worker:
        raise ValueError('Supply --worker user@spark-b on the first run')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', worker):
        raise ValueError('Invalid worker SSH target')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    if args.plan:
        print(f'Worker: {worker}\nDetect active RoCE links and check both hosts.\nCopy this recipe to the worker.\nBuild the pinned runtime and download/verify both models on each host.')
        print('Prepare only; leave services untouched.' if args.prepare_only else 'Start both ranks, then wait for the API to be ready.')
        print('Plan only: no SSH connections, downloads, writes, containers or GPU work.')
        return
    head = probe()
    remote = json.loads(ssh(worker, ['python3', '-', '--probe'], input=Path(__file__).read_bytes(), timeout=45))
    base = json.loads((ROOT / 'config.local.json' if (ROOT / 'config.local.json').exists() else ROOT / 'config.example.json').read_text())
    def saved_address(key):
        value = base[key]
        return None if value.startswith('192.0.2.') else value
    pairs = select_fabric(head['fabric'], remote['fabric'], args.head_address or saved_address('head_addr'),
                          args.worker_address or saved_address('worker_addr'))
    addresses = pairs[0][0]['address'], pairs[0][1]['address']
    for a, b in pairs:
        execute(['ping', '-c', '1', '-W', '2', '-I', a['interface'], b['address']])
        ssh(worker, ['ping', '-c', '1', '-W', '2', '-I', b['interface'], a['address']], timeout=10)
    data = args.data_dir or (None if (ROOT / 'config.local.json').exists() else str(Path(head['home']) / '.local/share/thinkingcap-qwen38'))
    worker_data = args.worker_data_dir or previous.get('worker_data_dir') or str(Path(remote['home']) / '.local/share/thinkingcap-qwen38')
    if not Path(worker_data).is_absolute():
        raise ValueError('--worker-data-dir must be absolute')
    recipe = str(Path(worker_data) / 'recipe')
    head_cfg = config_for(base, [a for a, _ in pairs], addresses, data)
    worker_cfg = config_for(base, [b for _, b in pairs], addresses, worker_data)
    print(f"Fabric: {addresses[0]} / {addresses[1]}; {len(pairs)} active rail(s)", flush=True)
    # Refuse occupied head GPUs before transferring or downloading anything.
    if not args.prepare_only:
        ensure_idle(head_cfg, 0)
    ssh(worker, ['python3', '-c', EXTRACT, recipe], input=recipe_archive())
    request = json.dumps({'config': worker_cfg, 'rank': 1, 'prepare_only': args.prepare_only}).encode()
    # Check the worker before either host begins large downloads.
    ssh(worker, ['python3', str(Path(recipe) / 'install.py'), '--node-check'], input=request)
    STATE.write_text(json.dumps({'worker': worker, 'worker_data_dir': worker_data, 'recipe': recipe}, indent=2) + '\n')
    print('Preparing head Spark...', flush=True)
    prepare_node(head_cfg, 0, args.prepare_only)
    print('Preparing worker Spark...', flush=True)
    ssh(worker, ['python3', str(Path(recipe) / 'install.py'), '--node-prepare'], input=request, capture=False)
    if args.prepare_only:
        print('Both Sparks are prepared. Run ./install.sh to start them.')
        return
    started = []
    try:
        row = json.loads(ssh(worker, ['python3', str(Path(recipe) / 'install.py'), '--node-start'], input=request))
        if row['created']:
            started.append((1, row['id']))
        row = start_node(head_cfg, 0)
        if row['created']:
            started.append((0, row['id']))
        print('Waiting for ThinkingCap to become ready...', flush=True)
        wait_ready(head_cfg, worker, recipe, args.timeout)
    except BaseException:
        print('Startup did not finish. Removing only containers created by this invocation; downloads and caches remain.', file=sys.stderr)
        for rank, container_id in reversed(started):
            try:
                if rank:
                    ssh(worker, ['python3', str(Path(recipe) / 'install.py'), '--node-remove', container_id])
                else:
                    remove_created(head_cfg, container_id)
            except Exception as exc:
                print(f'Could not clean up rank {rank}: {exc}', file=sys.stderr)
        raise
    print(f"Ready: http://{addresses[0]}:{head_cfg['port']}/v1\nModel: {head_cfg['served_model_name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help='SSH alias or user@hostname of the second Spark')
    parser.add_argument('--head-address', help='Choose a specific head fabric IPv4 address')
    parser.add_argument('--worker-address', help='Choose a specific worker fabric IPv4 address')
    parser.add_argument('--data-dir', help='Absolute directory for head models and cache')
    parser.add_argument('--worker-data-dir', help='Absolute directory for worker recipe, models and cache')
    parser.add_argument('--prepare-only', action='store_true', help='Download and build without starting GPU services')
    parser.add_argument('--plan', action='store_true', help='Print the workflow without making changes or connecting')
    parser.add_argument('--timeout', type=int, default=1800, help='API startup timeout in seconds (default: 1800)')
    for name in ('probe', 'node-check', 'node-prepare', 'node-start', 'node-status'):
        parser.add_argument('--' + name, action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--node-remove', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.probe:
            print(json.dumps(probe()))
        elif args.node_status:
            import serve
            cfg = serve.load(ROOT / 'config.local.json'); info = serve.container(cfg)
            print(json.dumps({'running': bool(info and info['State']['Running'])}))
        elif args.node_remove:
            import serve
            remove_created(serve.load(ROOT / 'config.local.json'), args.node_remove)
        elif args.node_check or args.node_prepare or args.node_start:
            import serve
            request = json.load(sys.stdin); cfg, rank = serve.validate(request['config']), request['rank']
            if rank not in (0, 1):
                raise ValueError('Rank must be 0 or 1')
            if args.node_check:
                if not request.get('prepare_only'):
                    ensure_idle(cfg, rank)
                if (ROOT / 'config.local.json').exists() and json.loads((ROOT / 'config.local.json').read_text()) != cfg:
                    raise RuntimeError('Worker config.local.json contains different settings; preserve or move it first')
            elif args.node_prepare:
                prepare_node(cfg, rank, request.get('prepare_only', False))
            else:
                print(json.dumps(start_node(cfg, rank)))
        else:
            if args.plan:
                install(args)
            else:
                # Keep simultaneous invocations on this head from racing over
                # configuration, downloads and the worker's deployment.
                with (ROOT / 'install.local.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError('Another installer is already running on this head')
                    install(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'Install stopped: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Install interrupted; run the same command to resume.\n')


if __name__ == '__main__':
    main()
