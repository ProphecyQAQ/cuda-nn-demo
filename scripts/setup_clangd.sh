#!/usr/bin/env bash
# Install clangd-23 (LLVM 23) for this project.
#
# Ubuntu 24.04 only ships clangd <= 20, and neither works here:
#   * clangd 18 does not know sm_120 (Blackwell, RTX 50xx);
#   * clangd 18/20 cannot parse the CUDA 13.x headers nvcc builds against
#     (they fail inside crt/math_functions.hpp and the removed
#     texture_fetch_functions.h).
# clangd 23 supports both CUDA 13.3 and sm_120.
#
# apt.llvm.org is often very slow from mainland China, so the TUNA mirror is
# used by default. Override with: LLVM_APT_MIRROR=https://apt.llvm.org/noble ...
set -euo pipefail

if command -v clangd-23 >/dev/null 2>&1; then
	echo "clangd-23 already installed:"
	clangd-23 --version | head -n1
	exit 0
fi

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
MIRROR="${LLVM_APT_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/llvm-apt}"

KEYRING=/etc/apt/keyrings/llvm-snapshot.gpg
LIST=/etc/apt/sources.list.d/llvm-23.list

if [ ! -s "$KEYRING" ]; then
	sudo mkdir -p "$(dirname "$KEYRING")"
	# The signing key is only published upstream (about 2 KB).
	curl -fsSL --retry 5 --retry-all-errors \
		https://apt.llvm.org/llvm-snapshot.gpg.key |
		sudo gpg --dearmor --yes -o "$KEYRING"
fi

echo "deb [signed-by=$KEYRING] ${MIRROR}/${CODENAME}/ llvm-toolchain-${CODENAME}-23 main" |
	sudo tee "$LIST" >/dev/null

sudo apt-get update
sudo apt-get install -y clangd-23

clangd-23 --version | head -n1