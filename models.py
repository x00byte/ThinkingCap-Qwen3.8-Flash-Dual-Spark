"""Verify the published target and draft manifests without loading either model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(directory, manifest, full=True):
    directory = Path(directory)
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path = directory / relative
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing or wrong-size model file: {path}')
        if not full:
            continue
        expected = record['sha256'] or record['git_blob']
        if not expected:
            raise ValueError(f'No hash recorded for {path}')
        digest = hashlib.sha256() if record['sha256'] else hashlib.sha1()
        if not record['sha256']:
            digest.update(f"blob {record['bytes']}\0".encode())
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Model hash mismatch: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for model in json.loads((ROOT / 'models.lock.json').read_text()):
        verify(args.root / model['repo'], model)
        print(f"Verified {model['repo']} at {model['revision']}")
