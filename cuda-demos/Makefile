# cuda-demos — 构建本目录下所有 demo
#
#   make                     编译全部 *.cu
#   make 05_hash_grid        只编译一个
#   make DEBUG=1             带 -g -G（配合 compute-sanitizer 用，很慢）
#   make ARCH=sm_89          换架构；ARCH=native 让 nvcc 自己识别本机 GPU
#   make list                看看有哪些 demo
#   make clean
#
# 默认 sm_120（RTX 50 系）。手搓单个 demo 时等价于：
#   nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo 05_hash_grid.cu -o 05_hash_grid

.DEFAULT_GOAL := all

# ---------------------------------------------------------------------------
# 工具链：nvcc 不在 PATH 时退回 /usr/local/cuda/bin
# ---------------------------------------------------------------------------
NVCC ?= nvcc
ifeq ($(shell command -v $(NVCC) 2>/dev/null),)
  NVCC := /usr/local/cuda/bin/nvcc
endif

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
ARCH  ?= sm_120
DEBUG ?= 0

NVFLAGS := -O3 -std=c++17 -arch=$(ARCH)

# -lineinfo 保留源码行号，给 nsys / ncu 看，几乎不影响性能；
# 但它和 -G 冲突（ptxas 会警告），所以只在 release 下加。
ifeq ($(DEBUG),1)
  NVFLAGS += -g -G      # -G 关掉优化 + 开设备端断言，越界才会被抓到
  SUFFIX  := _dbg
else
  NVFLAGS += -lineinfo
  SUFFIX  :=
endif

SRCS := $(wildcard *.cu)
BINS := $(addsuffix $(SUFFIX),$(SRCS:.cu=))

# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------
all: $(BINS)

# SUFFIX 为空时就是 "%: %.cu"；DEBUG=1 时是 "%_dbg: %.cu"
%$(SUFFIX): %.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# 改 Makefile 后全部重编一次，避免参数变了还用旧产物
$(BINS): Makefile

clean:
	rm -f $(SRCS:.cu=) $(addsuffix _dbg,$(SRCS:.cu=))

list:
	@echo "demos:"; for f in $(SRCS); do printf '  %s\n' "$$f"; done

help:
	@echo "usage:"
	@echo "  make                     build all *.cu"
	@echo "  make 01_vecadd_bandwidth build one (no .cu suffix)"
	@echo "  make DEBUG=1             with -g -G, for compute-sanitizer"
	@echo "  make ARCH=sm_89          override arch (ARCH=native = auto-detect)"
	@echo "  make list                list sources"
	@echo "  make clean"

.PHONY: all clean list help
