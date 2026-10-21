#!/usr/bin/env bash
# Fetch / verify the third-party dependencies of tiny-cuda-nn-demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TCNN_DIR="$ROOT/third_party/tiny-cuda-nn"
TCNN_REPO="https://github.com/NVLabs/tiny-cuda-nn.git"
# Pinned to the tiny-cuda-nn v2.0-era commit used by the demo.
TCNN_COMMIT="2e757bbe781db59c4980d389d7dccbf5edc09669"

if [ ! -d "$TCNN_DIR/.git" ]; then
    echo "==> cloning tiny-cuda-nn into third_party/"
    rm -rf "$TCNN_DIR"
    git clone "$TCNN_REPO" "$TCNN_DIR"
fi

echo "==> checking out ${TCNN_COMMIT:0:8}"
if ! git -C "$TCNN_DIR" cat-file -e "${TCNN_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$TCNN_DIR" fetch --tags --quiet
fi
git -C "$TCNN_DIR" checkout --quiet "$TCNN_COMMIT"
git -C "$TCNN_DIR" submodule update --init --recursive

for header in stb_image.h stb_image_write.h; do
    if [ ! -f "$ROOT/third_party/stb/$header" ]; then
        echo "error: missing third_party/stb/$header" >&2
        exit 1
    fi
done

echo "==> tiny-cuda-nn ready: $(git -C "$TCNN_DIR" rev-parse --short HEAD)"
