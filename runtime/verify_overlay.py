"""Check the source snapshot; optionally verify files installed in an image."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(directory):
    records = json.loads((ROOT / 'overlay-lock.json').read_text())
    for name, expected in records.items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Overlay hash mismatch: {name}')
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    print(f'Verified {len(records)} source files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', type=Path)
    args = parser.parse_args()
    verify(args.installed or ROOT / 'overlay')
