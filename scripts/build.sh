#!/usr/bin/env bash
# Configure and build tiny-cuda-nn-demo.
#
# Usage:
#   ./scripts/build.sh            # incremental build
#   ./scripts/build.sh --clean    # wipe build/ and rebuild from scratch
#
# Environment variables:
#   BUILD_DIR              build directory (default: <repo>/build)
#   BUILD_TYPE             CMake build type (default: Release)
#   JOBS                   parallel compile jobs (default: nproc)
#   TCNN_CUDA_ARCHITECTURES  force a GPU arch, e.g. 89 or "89;120"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$arg' (try --help)" >&2; exit 1 ;;
    esac
done

if ! command -v nvcc >/dev/null 2>&1 && [ -z "${CUDACXX:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
fi
if ! command -v nvcc >/dev/null 2>&1 && [ -z "${CUDACXX:-}" ]; then
    echo "error: nvcc not found. Install the CUDA toolkit or set CUDACXX." >&2
    exit 1
fi

bash "$ROOT/scripts/setup_deps.sh"

if [ "$CLEAN" -eq 1 ]; then
    echo "==> removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

CMAKE_ARGS=(
    -S "$ROOT"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_CUDA_COMPILER="${CUDACXX:-$(command -v nvcc)}"
)
if [ -n "${TCNN_CUDA_ARCHITECTURES:-}" ]; then
    CMAKE_ARGS+=("-DTCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES")
fi

echo "==> configuring ($BUILD_TYPE)"
cmake "${CMAKE_ARGS[@]}"

echo "==> building with $JOBS jobs"
cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo ""
echo "==> built: $BUILD_DIR/tcnn_demo"
echo "    run:   ./scripts/run.sh --help"
