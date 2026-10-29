"""Plan and operate one rank of the ThinkingCap server on the local host."""
from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess

from models import verify

ROOT = Path(__file__).resolve().parent
OWNER = 'x00byte.thinkingcap-qwen38'


def load(path):
    return validate(json.loads(Path(path).read_text()))


def validate(cfg):
    if cfg['profile'] != 'thinkingcap':
        raise ValueError('This runtime requires the thinkingcap profile')
    for key in ('model_dir', 'draft_dir', 'cache_dir'):
        value = cfg[key]
        if not Path(value).is_absolute() or ':' in value or '\n' in value:
            raise ValueError(f'{key} must be an absolute path without colons or newlines')
    for key in ('head_addr', 'worker_addr'):
        ipaddress.IPv4Address(cfg[key])
    if cfg['head_addr'] == cfg['worker_addr']:
        raise ValueError('The ranks require distinct fabric addresses')
    for key in ('port', 'master_port'):
        if type(cfg[key]) is not int or not 1024 <= cfg[key] <= 65535:
            raise ValueError(f'Invalid {key}')
    if cfg['port'] == cfg['master_port']:
        raise ValueError('API and master ports must differ')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', cfg['container_name']):
        raise ValueError('Invalid container name')
    if not cfg['image'] or cfg['image'].startswith('-'):
        raise ValueError('Invalid image reference')
    if not cfg['served_model_name'] or cfg['served_model_name'].startswith('-'):
        raise ValueError('Invalid served model name')
    return cfg


def overlay_fingerprint():
    expected = json.loads((ROOT / 'runtime/overlay-lock.json').read_text())
    return hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()


def config_fingerprint(cfg, rank):
    payload = {'config': cfg, 'rank': rank, 'overlay': overlay_fingerprint(),
               'profile': json.loads((ROOT / 'profiles/thinkingcap.json').read_text())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def command(cfg, rank):
    if rank not in (0, 1):
        raise ValueError('Rank must be 0 or 1')
    profile = json.loads((ROOT / 'profiles/thinkingcap.json').read_text())
    env = dict(profile['env'])
    env.update(VLLM_HOST_IP=cfg['worker_addr'] if rank else cfg['head_addr'],
               NCCL_SOCKET_IFNAME=cfg['fabric_interface'], GLOO_SOCKET_IFNAME=cfg['fabric_interface'],
               NCCL_IB_HCA=cfg['fabric_hca'], NCCL_IB_GID_INDEX=str(cfg['gid_index']),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_CACHE_ROOT='/cache/vllm-cache',
               FLASHINFER_WORKSPACE_BASE='/cache/flashinfer-workspace')
    args = ['docker', 'run', '-d', '--name', cfg['container_name'],
            '--label', 'org.thinkingcap.owner=' + OWNER,
            '--label', 'org.thinkingcap.config=' + config_fingerprint(cfg, rank),
            '--gpus', 'all', '--network', 'host', '--ipc', 'host',
            '--device', '/dev/infiniband', '--cap-add', 'IPC_LOCK',
            '--security-opt', 'seccomp=unconfined', '--ulimit', 'memlock=-1:-1',
            '--ulimit', 'stack=67108864', '--log-opt', 'max-size=200m',
            '--log-opt', 'max-file=3']
    if cfg.get('cpuset'):
        args += ['--cpuset-cpus', cfg['cpuset']]
    for key, target in [('model_dir', '/model:ro'), ('draft_dir', '/draft:ro'), ('cache_dir', '/cache')]:
        args += ['-v', cfg[key] + ':' + target]
    for key, value in sorted(env.items()):
        args += ['-e', f'{key}={value}']
    args += ['--entrypoint', 'vllm', cfg['image'], 'serve', '/model',
             '--host', '0.0.0.0', '--port', str(cfg['port']), '--nnodes', '2',
             '--master-addr', cfg['head_addr'], '--master-port', str(cfg['master_port']),
             '--node-rank', str(rank)]
    if rank:
        args += ['--headless']
    settings = dict(profile['vllm'], **{'served-model-name': cfg['served_model_name']})
    for key, value in settings.items():
        if value is True:
            args += ['--' + key]
        elif value is not False and value is not None:
            args += ['--' + key, str(value)]
    return args


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def container(cfg):
    ids = run(['docker', 'ps', '-aq', '--filter', 'name=^/' + cfg['container_name'] + '$'])
    if not ids:
        return None
    if len(ids.splitlines()) != 1:
        raise RuntimeError('Ambiguous container identity')
    return json.loads(run(['docker', 'inspect', ids]))[0]


def start(cfg, rank):
    if container(cfg):
        raise RuntimeError('Container already exists; inspect its logs before retrying')
    if run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
        raise RuntimeError('GPU is occupied; drain and stop the current service first')
    image = json.loads(run(['docker', 'image', 'inspect', cfg['image']]))[0]
    labels = image.get('Config', {}).get('Labels') or {}
    fingerprint = overlay_fingerprint()
    if image['Architecture'] != 'arm64' or labels.get('org.thinkingcap.overlay') != fingerprint:
        raise RuntimeError('Build the published ThinkingCap runtime before starting')
    for model, key in zip(json.loads((ROOT / 'models.lock.json').read_text()), ('model_dir', 'draft_dir')):
        verify(cfg[key], model, full=False)
    if not Path('/dev/infiniband').is_dir():
        raise RuntimeError('No /dev/infiniband devices found')
    if not (Path('/sys/class/net') / cfg['fabric_interface']).is_dir():
        raise RuntimeError('Fabric interface not found')
    for hca in cfg['fabric_hca'].split(','):
        state = Path('/sys/class/infiniband') / hca / 'ports/1/state'
        if not state.is_file() or 'ACTIVE' not in state.read_text():
            raise RuntimeError(f'RoCE HCA is not active: {hca}')
    address = cfg['worker_addr'] if rank else cfg['head_addr']
    if ipaddress.ip_address(address) in ipaddress.ip_network('192.0.2.0/24'):
        raise ValueError('Replace the example fabric addresses before starting')
    # Confirm this rank's address belongs to the local host.
    with socket.socket() as probe:
        probe.bind((address, 0))
    # Rank 0 serves HTTP. Rank 1's headless launcher does not bind that port.
    for port in ([cfg['port'], cfg['master_port']] if rank == 0 else []):
        with socket.socket() as probe:
            probe.bind(('0.0.0.0', port))
    cache = Path(cfg['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    profile = json.loads((ROOT / 'profiles/thinkingcap.json').read_text())
    (cache / 'mbx-depth.json').write_text(json.dumps(profile['mtp_depth']) + '\n')
    print(run(command(cfg, rank)))
    print('Container created. Wait for /health on rank 0 before sending requests.')


def stop(cfg):
    info = container(cfg)
    if info is None:
        print('No project container exists.')
        return
    labels = info.get('Config', {}).get('Labels') or {}
    if labels.get('org.thinkingcap.owner') != OWNER:
        raise RuntimeError('Refusing to remove a container without this project’s ownership label')
    # Use the inspected immutable ID so a name change cannot target another container.
    print(run(['docker', 'rm', '-f', info['Id']]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'start', 'status', 'stop'])
    parser.add_argument('--rank', type=int, choices=[0, 1])
    parser.add_argument('--config', type=Path, default=ROOT / 'config.local.json')
    args = parser.parse_args()
    try:
        cfg = load(args.config)
        if args.action in ('plan', 'start') and args.rank is None:
            parser.error('--rank is required for plan and start')
        if args.action == 'plan':
            print(shlex.join(command(cfg, args.rank)))
        elif args.action == 'start':
            start(cfg, args.rank)
        elif args.action == 'stop':
            stop(cfg)
        else:
            info = container(cfg)
            print(json.dumps({'name': cfg['container_name'], 'state': info['State'] if info else 'absent'}, indent=2))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')
