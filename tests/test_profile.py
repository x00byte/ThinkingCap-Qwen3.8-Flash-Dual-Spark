import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import models
import serve


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = serve.load(serve.ROOT / 'config.example.json')

    def test_rank_commands_preserve_serving_contract(self):
        for rank in (0, 1):
            args = serve.command(self.cfg, rank)
            self.assertEqual(args[args.index('--node-rank') + 1], str(rank))
            self.assertEqual('--headless' in args, bool(rank))
            self.assertEqual(args[args.index('--max-model-len') + 1], '262144')
            self.assertEqual(args[args.index('--tensor-parallel-size') + 1], '2')
            self.assertEqual(args[args.index('--max-num-seqs') + 1], '16')
            self.assertNotIn('--hf-overrides', args)
            self.assertNotIn('--async-scheduling', args)
            self.assertIn('THINKINGCAP_DFLASH_EXTEND_CONTEXT=0', args)
            self.assertIn('MBX_PLE_PREFILL_CKPT=1', args)
            self.assertIn('VLLM_HOST_IP=' + self.cfg['worker_addr' if rank else 'head_addr'], args)
            spec = json.loads(args[args.index('--speculative-config') + 1])
            self.assertEqual((spec['method'], spec['num_speculative_tokens']), ('dflash', 3))

    def test_occupied_gpu_never_starts_or_stops_a_container(self):
        calls = []
        def fake(args):
            calls.append(args)
            return '' if args[:2] == ['docker', 'ps'] else '1234'
        with patch.object(serve, 'run', side_effect=fake):
            with self.assertRaisesRegex(RuntimeError, 'GPU is occupied'):
                serve.start(self.cfg, 0)
        self.assertEqual(len(calls), 2)
        self.assertFalse(any('rm' in c or 'run' in c for c in calls))

    def test_existing_container_blocks_start(self):
        with patch.object(serve, 'container', return_value={'Id': 'existing'}), patch.object(serve, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                serve.start(self.cfg, 1)
            run.assert_not_called()

    def test_stop_rejects_unowned_container(self):
        with patch.object(serve, 'container', return_value={'Config': {'Labels': {}}}), patch.object(serve, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'ownership label'):
                serve.stop(self.cfg)
            run.assert_not_called()

    def test_stop_uses_inspected_id(self):
        info = {'Id': 'sha256:owned', 'Config': {'Labels': {'org.thinkingcap.owner': serve.OWNER}}}
        with patch.object(serve, 'container', return_value=info), patch.object(serve, 'run', return_value='removed') as run:
            serve.stop(self.cfg)
            run.assert_called_once_with(['docker', 'rm', '-f', 'sha256:owned'])

    def test_invalid_paths_fail_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(self.cfg, model_dir='/models:/injected')
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'absolute path'):
                serve.load(path)


class ModelTests(unittest.TestCase):
    def test_hash_check_catches_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.bin'
            path.write_bytes(b'good')
            record = {'path': 'model.bin', 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest(), 'git_blob': None}
            manifest = {'files': [record]}
            models.verify(directory, manifest)
            path.write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                models.verify(directory, manifest)

    def test_git_blob_hash_includes_header(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_bytes(b'{}')
            manifest = {'files': [{'path': 'config.json', 'bytes': 2, 'sha256': None,
                                  'git_blob': hashlib.sha1(b'blob 2\0{}').hexdigest()}]}
            models.verify(directory, manifest)

    def test_manifest_cannot_escape_model_directory(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe manifest path'):
            models.verify('/srv/models', {'files': [{'path': '../other'}]})


if __name__ == '__main__':
    unittest.main()
