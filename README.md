# tiny-cuda-nn-demo

一个基于 [tiny-cuda-nn](https://github.com/NVLabs/tiny-cuda-nn) 的最小 C++/CUDA 示例工程：
用 **HashGrid 编码 + 全连接 MLP** 拟合一张 2D 图像（输入 `(u, v)`，输出 `RGB`），
并用 [stb_image](https://github.com/nothings/stb) 读取输入图、写出参考图与训练结果。

## 功能

- **开箱即用**：不传图片时自动生成一张彩色测试图（色轮 + 棋盘 + 网格），无需准备数据。
- **在线采样**：参考图只上传 GPU 一次，每个训练 step 在 CUDA kernel 中双线性采样，
  不会把逐像素训练集读进内存。
- **可视化训练过程**：每 `save_every` 步输出一张预览图到 `output/previews/`。
- **结果对比**：训练结束写出 `learned.png`、`comparison.png`（左参考图 / 右学习结果），并打印 PSNR。
- **JIT 融合**：默认尝试开启 tiny-cuda-nn 的 JIT fusion；如果驱动版本低于 CUDA runtime
  （PTX 版本不兼容），会自动关闭并说明原因，避免第一次训练时报错。

## 目录结构

```
tiny-cuda-nn-demo/
├── CMakeLists.txt              # 构建入口
├── README.md
├── .gitignore
├── configs/
│   └── image_hashgrid.json     # HashGrid + MLP + Adam 配置
├── scripts/
│   ├── setup_deps.sh           # 拉取 / 校验 tiny-cuda-nn
│   ├── build.sh                # 配置 + 编译
│   └── run.sh                  # 运行
├── src/
│   └── main.cpp                # 唯一源文件：stb 读写、CLI、采样 kernel、训练、推理
├── third_party/
│   ├── stb/                    # stb_image.h + stb_image_write.h（已内置）
│   └── tiny-cuda-nn/           # 由 scripts/setup_deps.sh 拉取（固定 commit）
└── output/                     # 运行产物
```

整个工程只有 `src/main.cpp` 一个源文件（没有 `.cu`）。该文件内含一个双线性采样 kernel，
因此 CMake 通过 `set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)` 让 nvcc
编译它；同时 tiny-cuda-nn 的模板（`network_precision_t`）也只在 nvcc 下才可用。

## 环境要求

| 依赖 | 版本 |
| --- | --- |
| CMake | >= 3.24 |
| CUDA Toolkit | >= 12.8（`sm_120` 需要 12.8+） |
| GCC / Clang | 支持 C++17（GCC 13 已验证） |
| GPU | NVIDIA，显存 >= 2 GB |
| 操作系统 | Linux（WSL2 已验证）、Windows 也可用 |

本工程验证环境：WSL2 Ubuntu 24.04 + RTX 5070（sm_120）+ CUDA 13.4 + 驱动 591.59。

## 快速开始

```bash
cd /root/neural-render/tiny-cuda-nn-demo

./scripts/setup_deps.sh   # 首次运行：clone tiny-cuda-nn（含 cutlass/fmt/cmrc 子模块）
./scripts/build.sh        # 配置 + 编译，产物在 build/tcnn_demo
./scripts/run.sh          # 默认：测试图，2000 步，batch 65536
```

首次编译需要编译 tiny-cuda-nn 的全部 CUDA 源文件：24 核机器约 2~3 分钟，普通笔记本约 5~15 分钟；
之后的增量编译只需要十几秒。

常用变体：

```bash
# 用 512x512 测试图训练 5000 步，每 500 步存一张预览
./scripts/run.sh -n 5000 --save-every 500

# 学习自己的图片
./scripts/run.sh -i data/my_photo.jpg -o output/my_photo.png -n 5000

# 干净重编
./scripts/build.sh --clean

# 指定 GPU 和随机种子
./scripts/run.sh --gpu 0 --seed 42
```

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-i, --input PATH` | 要拟合的图片（PNG/JPG/BMP/TGA） | 无 → 生成测试图 |
| `-o, --output PATH` | 最终输出图片，按扩展名决定格式 | `output/learned.png` |
| `-c, --config PATH` | JSON 配置文件 | `configs/image_hashgrid.json` |
| `-n, --steps N` | 训练步数 | 配置文件 `train.n_steps`（2000） |
| `-b, --batch-size N` | 每步采样数（自动向上取整到 256 的倍数） | 65536 |
| `--save-every N` | 每 N 步存一张预览，0 表示关闭 | 250 |
| `--seed N` | 随机种子 | 1337 |
| `--gpu N` | CUDA 设备序号 | 0 |
| `--width N` / `--height N` | 测试图尺寸（仅在没有 `--input` 时生效） | 512 × 512 |
| `--gamma G` | 写图片时的显示 gamma（`pow(v, 1/G)`） | 1.0 |
| `--no-jit` | 关闭 tiny-cuda-nn 的 JIT fusion | 不关闭 |
| `-h, --help` | 帮助 | — |

## 配置文件

`configs/image_hashgrid.json` 的 `loss` / `optimizer` / `encoding` / `network`
会原样传给 tiny-cuda-nn，具体字段见其
[DOCUMENTATION.md](https://github.com/NVLabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md)。
`train` 段由本 demo 自己读取：

| 字段 | 说明 | 默认 |
| --- | --- | --- |
| `batch_size` | 每步样本数 | 65536 |
| `n_steps` | 训练步数 | 2000 |
| `save_every` | 预览间隔，0 = 关闭 | 250 |
| `seed` | 随机种子 | 1337 |

配置文件支持 `//` 注释。想提高拟合质量，可以增加 `n_steps`、`n_levels` 或
`log2_hashmap_size`；想更快出图，就减少 `n_steps` 和 `n_neurons`。

## 输出

运行结束后（默认 `output/`）会得到：

| 文件 | 说明 |
| --- | --- |
| `reference.png` | 参考图（输入图或生成的测试图） |
| `learned.png` | 网络拟合结果 |
| `comparison.png` | 左：参考图，右：拟合结果 |
| `previews/step_XXXXXX.png` | 训练过程中的中间结果 |

终端会打印每一步组的 loss、速度，以及最终 PSNR。

## 性能参考

RTX 5070（sm_120，JIT fusion 因驱动版本自动关闭）。GPU 反向传播的原子加导致训练不完全确定，
同一配置多次运行 PSNR 会有约 1~2 dB 波动：

| 步数 | 耗时 | 最终 PSNR |
| --- | --- | --- |
| 2000（默认） | ~1.8 s（约 1100 steps/s） | 约 39~41 dB |
| 5000 | ~4.3 s | 约 44 dB |

## 手动构建（不使用脚本）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --parallel "$(nproc)"
./build/tcnn_demo --help
```

CMake 会自动检测 GPU 架构（本机为 `sm_120`）。如需交叉编译或固定架构：

```bash
cmake -S . -B build -DTCNN_CUDA_ARCHITECTURES=89   # 或 "89;120"
```

> 注意：tiny-cuda-nn 会用自己的自动检测结果覆盖 `CMAKE_CUDA_ARCHITECTURES`，
> 所以请通过 `TCNN_CUDA_ARCHITECTURES` 指定架构，本工程会同步给自己的 kernel。

## 常见问题

**找不到 nvcc**

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
```

`scripts/build.sh` 也会自动尝试这个路径。

**JIT fusion 显示 off / 出现 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`**

JIT 用 NVRTC 在运行时编译 kernel，生成的 PTX 版本由 CUDA Toolkit 决定；如果驱动比
Toolkit 旧（例如驱动 591.59 对应 CUDA 13.1，而 Toolkit 是 13.4），驱动会拒绝加载。
本 demo 会在启动时检测并自动关闭 JIT（训练仍然正常，速度略低）。升级驱动即可开启，
也可以手动加 `--no-jit` 消除提示。

**为什么 main.cpp 需要 nvcc 编译**

tiny-cuda-nn 是纯 CUDA 库，`network_precision_t` 等类型只在 `__CUDACC__` 下定义；而且
`main.cpp` 里有一个采样 kernel。所以 CMake 把这个 `.cpp` 标记为 CUDA 源文件，工程中不需要
单独的 `.cu` 文件。

**显存不足**

减小 batch size，例如 `-b 16384`。

**CUDA 版本 / 架构不匹配**

`nvcc` 版本需要支持目标 GPU（`sm_120` 需要 CUDA >= 12.8）。
显式指定架构重编：`TCNN_CUDA_ARCHITECTURES=120 ./scripts/build.sh --clean`。

**编译很慢**

tiny-cuda-nn 的 CUDA 源文件较多，首次编译慢是正常的。`build.sh` 默认使用
`nproc` 个并行任务，可用 `JOBS=8 ./scripts/build.sh` 限制并行度。

**想换成自己的图片**

```bash
./scripts/run.sh -i /path/to/photo.png -o output/photo.png -n 5000
```

输入图会被转成 RGBA float 上传 GPU，训练时双线性采样；输出图保持相同分辨率。

## 依赖版本

| 依赖 | 版本 / commit |
| --- | --- |
| tiny-cuda-nn | `2e757bbe781db59c4980d389d7dccbf5edc09669`（v2.0 + 32 commits） |
| stb | `a7d4c2175630e1ea5b6f8dd00ab96fed0b97b96d` |
| CUDA | 13.4（验证） |

`third_party/tiny-cuda-nn` 不纳入版本控制，由 `scripts/setup_deps.sh` 拉取并固定到上述 commit；
`third_party/stb` 的两个头文件直接内置（见 `third_party/stb/README.md`）。

## 许可

- tiny-cuda-nn：BSD-3-Clause，见 `third_party/tiny-cuda-nn/LICENSE.txt`
- stb：public domain / MIT，见各头文件末尾