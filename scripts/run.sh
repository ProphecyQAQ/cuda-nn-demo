#!/usr/bin/env bash
# Run the demo from the repository root (so relative output paths work).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/tcnn_demo}"

if [ ! -x "$BIN" ]; then
    echo "error: '$BIN' not found. Build first with ./scripts/build.sh" >&2
    exit 1
fi

cd "$ROOT"
exec "$BIN" "$@"
