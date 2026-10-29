#!/usr/bin/env bash
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
    echo 'Python 3.10 or newer is required.' >&2
    exit 1
fi
exec python3 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/install.py" "$@"
