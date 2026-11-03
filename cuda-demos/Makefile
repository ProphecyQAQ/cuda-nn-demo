# cuda-demos — 构建本目录下所有 demo
#
#   make                     编译全部 *.cu（产物在 build/）
#   make vecadd              只编译一个（等价于 make build/vecadd）
#   make DEBUG=1             带 -g -G（配合 compute-sanitizer 用，很慢）
#   make ARCH=sm_89          换架构；ARCH=native 让 nvcc 自己识别本机 GPU
#   make list                看看有哪些 demo
#   make clean               删除整个 build/
#
# 默认 sm_120（RTX 50 系）。手搓单个 demo 时等价于：
#   nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo vecadd.cu -o build/vecadd

.DEFAULT_GOAL := all

BUILD_DIR := build

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

SRCS  := $(wildcard *.cu)
DEMOS := $(SRCS:.cu=)
BINS  := $(addprefix $(BUILD_DIR)/,$(addsuffix $(SUFFIX),$(DEMOS)))

# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------
all: $(BINS)

$(BUILD_DIR):
	mkdir -p $@

# build/foo: foo.cu
$(BUILD_DIR)/%$(SUFFIX): %.cu | $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $< -o $@

# 便利写法：make foo 等价于 make build/foo
$(addsuffix $(SUFFIX),$(DEMOS)): %: $(BUILD_DIR)/%
.PHONY: $(addsuffix $(SUFFIX),$(DEMOS))

# 改 Makefile 后全部重编一次，避免参数变了还用旧产物
$(BINS): Makefile

clean:
	rm -rf $(BUILD_DIR)

list:
	@echo "demos:"; for f in $(SRCS); do printf '  %s\n' "$$f"; done

help:
	@echo "usage:"
	@echo "  make                     build all *.cu into $(BUILD_DIR)/"
	@echo "  make vecadd              build one (no .cu suffix)"
	@echo "  make DEBUG=1             with -g -G, for compute-sanitizer"
	@echo "  make ARCH=sm_89          override arch (ARCH=native = auto-detect)"
	@echo "  make list                list sources"
	@echo "  make clean               remove $(BUILD_DIR)/"

.PHONY: all clean list help