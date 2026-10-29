import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import install
import serve


def rail(addr, hca='rdma0', interface='fabric0', gid='3'):
    return dict(address=addr, prefixlen=24, interface=interface, hca=hca, gid_index=gid)


class FabricTests(unittest.TestCase):
    def test_active_ipv4_roce_gids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory) / 'rdma0/ports/1'
            for name in ('gids', 'gid_attrs/ndevs', 'gid_attrs/types'):
                (port / name).mkdir(parents=True)
            (port / 'state').write_text('4: ACTIVE')
            for index, value, kind in [('1', 'fe80::1', 'RoCE v2'),
                                        ('2', '::ffff:10.20.0.1', 'IB/RoCE v1'),
                                        ('3', '::ffff:10.20.0.1', 'RoCE v2')]:
                (port / 'gids' / index).write_text(value)
                (port / 'gid_attrs/types' / index).write_text(kind)
                (port / 'gid_attrs/ndevs' / index).write_text('fabric0')
            addresses = [{'ifname': 'fabric0', 'addr_info': [dict(family='inet', local='10.20.0.1', prefixlen=24)]}]
            self.assertEqual(install.discover_fabric(Path(directory), addresses), [rail('10.20.0.1')])
            (port / 'state').write_text('1: DOWN')
            self.assertEqual(install.discover_fabric(Path(directory), addresses), [])

    def test_dual_rail_matches_both_hosts_and_gid_indices(self):
        head = [rail('10.20.0.1'), rail('10.30.0.1', 'rdma1', 'fabric1')]
        worker = [rail('10.20.0.2', gid='5'), rail('10.30.0.2', 'rdma1', 'fabric1', gid='5')]
        selected = install.select_fabric(head, worker)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0][0]['address'], '10.20.0.1')
        self.assertEqual(selected[0][1]['gid_index'], '5')
        worker[1]['gid_index'] = '7'
        self.assertEqual(len(install.select_fabric(head, worker)), 1)

    def test_explicit_addresses_and_unmatched_subnet(self):
        head = [rail('10.20.0.1'), rail('10.30.0.1', 'rdma1', 'fabric1')]
        worker = [rail('10.20.0.2'), rail('10.30.0.2', 'rdma1', 'fabric1')]
        self.assertEqual(install.select_fabric(head, worker, '10.30.0.1')[0][0]['hca'], 'rdma1')
        with self.assertRaisesRegex(RuntimeError, 'No shared RoCE subnet'):
            install.select_fabric(head, [rail('10.40.0.2')])


class TransferTests(unittest.TestCase):
    def archive(self, name, data=b'file'):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as archive:
            member = tarfile.TarInfo(name); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        return output.getvalue()

    def extract(self, root, data):
        return subprocess.run([sys.executable, '-c', install.EXTRACT, str(root)],
                              input=data, capture_output=True)

    def test_transfer_excludes_credentials_and_local_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'install.py').write_text('print("installer")')
            (root / 'config.local.json').write_text('PRIVATE_CONFIG')
            (root / '.git').mkdir(); (root / '.git/config').write_text('PRIVATE_REMOTE')
            with patch.object(install, 'ROOT', root), patch.object(install, 'TRANSFER_ROOTS', ['install.py']):
                archive = install.recipe_archive()
            self.assertNotIn(b'PRIVATE_CONFIG', archive)
            self.assertNotIn(b'PRIVATE_REMOTE', archive)
            worker = root / 'worker'
            self.assertEqual(self.extract(worker, archive).returncode, 0)
            self.assertEqual(self.extract(worker, archive).returncode, 0)
            self.assertEqual((worker / 'install.py').read_text(), 'print("installer")')

    def test_extraction_rejects_traversal_and_unowned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertNotEqual(self.extract(root / 'recipe', self.archive('../escape')).returncode, 0)
            self.assertFalse((root / 'escape').exists())
            unowned = root / 'unowned'; unowned.mkdir(); (unowned / 'notes').write_text('keep')
            self.assertNotEqual(self.extract(unowned, self.archive('notes', b'overwrite')).returncode, 0)
            self.assertEqual((unowned / 'notes').read_text(), 'keep')

    def test_extraction_rejects_existing_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); recipe = root / 'recipe'; recipe.mkdir()
            (recipe / '.thinkingcap-installer').write_text('x00byte.thinkingcap-qwen38')
            outside = root / 'outside'; outside.mkdir()
            (recipe / 'linked').symlink_to(outside, target_is_directory=True)
            self.assertNotEqual(self.extract(recipe, self.archive('linked/file')).returncode, 0)
            self.assertFalse((outside / 'file').exists())


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = serve.load(serve.ROOT / 'config.example.json')

    def test_settings_signature_changes_with_rank_and_profile(self):
        signature = serve.config_fingerprint(self.cfg, 0)
        self.assertNotEqual(signature, serve.config_fingerprint(self.cfg, 1))
        self.assertNotEqual(signature, serve.config_fingerprint(dict(self.cfg, port=8890), 0))
        args = serve.command(self.cfg, 0)
        self.assertIn('org.thinkingcap.config=' + signature, args)

    def test_idle_guard_rejects_occupied_gpu(self):
        with patch.object(install, 'running_service', return_value=None), patch.object(install, 'execute', return_value='42'):
            with self.assertRaisesRegex(RuntimeError, 'GPU is occupied'):
                install.ensure_idle(self.cfg, 0)

    def test_matching_service_is_reused_without_download_or_start(self):
        with patch.object(install, 'running_service', return_value={'Id': 'owned'}), patch.object(install, 'execute') as run:
            install.prepare_node(self.cfg, 0)
            self.assertEqual(install.start_node(self.cfg, 0), {'created': False, 'id': 'owned'})
            run.assert_not_called()

    def test_cleanup_removes_only_confirmed_owned_id(self):
        info = {'Id': 'owned', 'Config': {'Labels': {'org.thinkingcap.owner': serve.OWNER}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(serve, 'container', return_value=info), patch.object(serve, 'run') as run, patch.object(subprocess, 'run', return_value=argparse.Namespace(stdout='startup log', stderr='')):
            cfg = dict(self.cfg, cache_dir=directory)
            install.remove_created(cfg, 'other')
            run.assert_not_called()
            install.remove_created(cfg, 'owned')
            run.assert_called_once_with(['docker', 'rm', '-f', 'owned'])
            self.assertEqual((Path(directory) / 'install-startup.log').read_text(), 'startup log')

    def test_existing_config_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(install, 'ROOT', Path(directory)):
            install.save_config(self.cfg)
            with self.assertRaisesRegex(RuntimeError, 'different settings'):
                install.save_config(dict(self.cfg, port=8890))
            self.assertEqual(json.loads((Path(directory) / 'config.local.json').read_text()), self.cfg)

    def test_plan_makes_no_connections_or_writes(self):
        args = argparse.Namespace(worker='spark-b', timeout=1800, plan=True, prepare_only=False)
        with tempfile.TemporaryDirectory() as directory, patch.object(install, 'STATE', Path(directory) / 'state.json'), patch.object(install, 'execute') as run:
            install.install(args)
            run.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_concurrent_install_is_refused_before_host_checks(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'install.py').write_bytes(Path(install.__file__).read_bytes())
            with (root / 'install.local.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run([sys.executable, str(root / 'install.py'), '--worker', 'spark-b'],
                                        capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Another installer is already running', result.stderr)

    def test_invalid_worker_never_reaches_ssh(self):
        with patch.object(install, 'execute') as run:
            with self.assertRaises(ValueError):
                install.ssh('-oProxyCommand=bad', ['true'])
            run.assert_not_called()


class PreparationTests(unittest.TestCase):
    def test_corrupt_file_is_redownloaded_and_correct_files_are_kept(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = serve.load(serve.ROOT / 'config.example.json')
            cfg.update(model_dir=str(root / 'target'), draft_dir=str(root / 'draft'), cache_dir=str(root / 'cache'))
            models = []
            for name, key in [('target', 'model_dir'), ('draft', 'draft_dir')]:
                folder = Path(cfg[key]); folder.mkdir()
                (folder / 'weights').write_bytes(b'good')
                models.append({'repo': name, 'revision': 'pinned', 'files': [
                    {'path': 'weights', 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest(), 'git_blob': None}]})
            (root / 'target/weights').write_bytes(b'bad!')
            (root / 'models.lock.json').write_text(json.dumps(models))
            calls = []
            def fake_execute(args, **kwargs):
                calls.append(args)
                if 'pinned' in args:
                    (root / 'target/weights').write_bytes(b'good')
                return ''
            with patch.object(install, 'ROOT', root), patch.object(install, 'running_service', return_value=None), patch.object(install, 'ensure_idle'), patch.object(install, 'image_ready', return_value=True), patch.object(install, 'free_space'), patch.object(install, 'execute', side_effect=fake_execute):
                install.prepare_node(cfg, 0)
            downloads = [args for args in calls if 'pinned' in args]
            self.assertEqual(len(downloads), 1)
            self.assertEqual(downloads[0][-3:], ['pinned', '["weights"]', '1'])
            self.assertEqual((root / 'draft/weights').read_bytes(), b'good')


class OrchestrationTests(unittest.TestCase):
    def run_failure(self, worker_created, head_started):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.example.json').write_bytes((serve.ROOT / 'config.example.json').read_bytes())
            args = argparse.Namespace(worker='spark-b', timeout=10, plan=False, prepare_only=False,
                                      head_address=None, worker_address=None, data_dir=None, worker_data_dir=None)
            head = {'home': str(root / 'head'), 'fabric': [rail('10.20.0.1')]}
            remote = {'home': str(root / 'worker'), 'fabric': [rail('10.20.0.2')]}
            calls = []
            def fake_ssh(worker, command, **kwargs):
                calls.append(command)
                if '--probe' in command:return json.dumps(remote)
                if '--node-start' in command:return json.dumps({'created': worker_created, 'id': 'worker-id'})
                return ''
            start = {'created': True, 'id': 'head-id'} if head_started else RuntimeError('head start failed')
            with patch.object(install, 'ROOT', root), patch.object(install, 'STATE', root / 'cluster.local.json'), patch.object(install, 'probe', return_value=head), patch.object(install, 'ssh', side_effect=fake_ssh), patch.object(install, 'execute'), patch.object(install, 'ensure_idle'), patch.object(install, 'recipe_archive', return_value=b'archive'), patch.object(install, 'prepare_node'), patch.object(install, 'start_node', **({'return_value': start} if head_started else {'side_effect': start})), patch.object(install, 'wait_ready', side_effect=TimeoutError('not ready')), patch.object(install, 'remove_created') as cleanup:
                with self.assertRaises((RuntimeError, TimeoutError)):
                    install.install(args)
            return calls, cleanup.call_args_list

    def test_head_failure_cleans_up_new_worker(self):
        calls, local = self.run_failure(worker_created=True, head_started=False)
        self.assertTrue(any('--node-remove' in args and args[-1] == 'worker-id' for args in calls))
        self.assertEqual(local, [])

    def test_failure_preserves_preexisting_worker(self):
        calls, local = self.run_failure(worker_created=False, head_started=False)
        self.assertFalse(any('--node-remove' in args for args in calls))
        self.assertEqual(local, [])

    def test_timeout_cleans_up_only_new_ranks(self):
        calls, local = self.run_failure(worker_created=False, head_started=True)
        self.assertFalse(any('--node-remove' in args for args in calls))
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0].args[1], 'head-id')


if __name__ == '__main__':
    unittest.main()
