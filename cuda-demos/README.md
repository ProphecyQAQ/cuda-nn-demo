# CUDA Hash Grid 学习计划

从「会写 vec add」到「独立实现并接入 tiny-cuda-nn 的 hash grid encoding」。

- 环境：WSL2 Ubuntu 24.04 / RTX 5080 (sm_120) / CUDA 13.3
- 起点：`src/encoding.h` 已有类骨架（构造、参数、`Encoding<T>` 接口），`forward_impl` / `backward_impl` 待实现
- 目标：先 2D，再推广到 3D

---

## 1. 知识地图

### 1.1 必补的 CUDA 基础

| 知识 | 在 hash grid 里用在哪 | Demo |
| --- | --- | --- |
| grid / block 索引、grid-stride loop | 所有 kernel | 01 |
| 合并访存、内存带宽 | 特征表读写、输入输出 | 01, 02 |
| shared memory、`__syncthreads()` | 转置、tile、warp 协作 | 02 |
| 2D 索引、边界 clamp | 网格坐标、角点 | 02, 03 |
| 双线性 / 三线性插值 | forward 核心 | 03 |
| 多 level、拼接、对齐 | 编码输出 | 04 |
| hash 函数、模数、碰撞 | 稀疏存储 | 05 |
| `__half` / `__half2`、向量化 load | 参数与输出 | 06 |
| `atomicAdd`、竞争 | 反向梯度散射 | 07, 08 |
| 有限差分 | 梯度验证 | 07 |
| stream / 异步 / 错误检查 | 接入 tcnn | 09 |
| ncu / nsys / compute-sanitizer | 性能与越界排查 | 全程 |

### 1.2 Hash grid 的数学

1. **顶点 vs 格子**：`res` 个顶点夹出 `res-1` 个格子；特征存在顶点上，坐标范围是 `[0, res-1]`。
2. **每层分辨率**：`res_l = ceil(base * per_level_scale^l)`（tcnn 的顶点约定）。
3. **坐标映射**：`pos = fma(scale, x, 0.5)`，其中 `scale = base * per_level_scale^l - 1`；`ix = floor(pos)`，`fx = pos - ix`。
4. **插值**：`2^d` 个角点，权重 `w = prod(frac or 1-frac)`。
5. **每层存储**：`table_size = min(res^d, T)`，向上对齐到 8，offset 做前缀和。
6. **hash**：`idx = (dense < size) ? dense : hash(cell) % size`。
7. **碰撞率**：约 `1 - exp(-res^d / T)`。

### 1.3 训练与集成

1. 反向链式法则：`dL/dF += w * dL/dy`；`dL/dx = sum(dL/dy * F * dw/dx)`。
2. `GradientMode::Overwrite` 需要先清零梯度。
3. fp16 梯度要用 float 缓冲或 `half2` 原子加。
4. 接入 `tcnn::Encoding<T>`：`forward_impl` / `backward_impl` / padding / `preferred_output_layout`。
5. `register_encoding` 让 JSON 的 `"otype"` 找到你的类。

---

## 2. Demo 系列

```
cuda-demos/
  01_vecadd_bandwidth.cu
  02_transpose.cu
  03_bilinear_dense.cu
  04_multilevel_dense.cu
  05_hash_grid.cu
  06_half2.cu
  07_backward_dense.cu
  08_backward_hash.cu
  09_tcnn_integration.cu
  10_optimize.cu
```

编译：

```bash
nvcc -arch=sm_120 -O3 -std=c++17 demo.cu -o demo
```

| # | Demo | 学到什么 | 验收标准 | 时间 |
| --- | --- | --- | --- | --- |
| 01 | vec add + 带宽 | grid-stride、`float4`、计时 | 实测带宽接近 700 GB/s | 0.5 天 |
| 02 | 矩阵转置 | shared memory、合并访存、AoS/SoA | 比 naive 快 2~3 倍 | 1 天 |
| 03 | 单层 dense 双线性 | 顶点/格子、4 角点、权重 | 与 CPU 误差 < 1e-6 | 1 天 |
| 04 | 多 level dense | 每层参数、拼接、`__constant__` | 与 CPU 逐元素一致 | 1 天 |
| 05 | 换成 hash | hash 函数、模数、碰撞 | `res^2 <= T` 时与 dense 完全相等 | 1 天 |
| 06 | half + 向量化 | `__half`、`half2`、对齐 | 与 float 版误差 < 1e-3 | 1 天 |
| 07 | dense 反向 | `atomicAdd`、有限差分 | 梯度相对误差 < 1e-3 | 1~2 天 |
| 08 | hash 反向 | 原子竞争、float 累积 | 与 dense 梯度一致 | 1 天 |
| 09 | 接 tcnn | Encoding 接口、stream | 训练出可辨认图像 | 1 天 |
| 10 | 优化 | ncu、L2、排序、shared 暂存 | 比 09 快 2 倍以上 | 2 天+ |

---

## 3. Demo 01：vec add 与带宽

**目标**：建立「带宽 = 字节数 / 时间」的直觉，学会用 `cudaEvent` 正确计时。

```cpp
__global__ void add(const float4* a, const float4* b, float4* c, size_t n4) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += gridDim.x * blockDim.x) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}
```

要点：

- 数据量 >= 256 MB，否则测的是 L2 而不是显存。
- 先 warmup 再计时，重复 10 次取平均。
- 对比 `float` 与 `float4` 版本。
- 验收：`(3 * N * 4 bytes) / time` 接近显卡峰值带宽（RTX 5080 理论约 960 GB/s，实测 700~800）。

---

## 4. Demo 02：矩阵转置

**目标**：理解「读写方向不一致」导致的非合并访存——这正是 AoS / SoA 的物理原因。

naive：

```cpp
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    out[x * n + y] = in[y * n + x];   // 写不合并
}
```

tiled：

```cpp
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[32][33];    // +1 避免 bank conflict
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}
```

验收：tiled 比 naive 快 2~3 倍；用 compute-sanitizer 确认无越界。

---

## 5. Demo 03：单层 dense 双线性（核心）

**目标**：把 hash grid 的数学跑通，**先不碰 hash**。

```cpp
// table: res*res 个顶点，每个 F=2 维特征（AoS）
__global__ void bilinear(
    const float2* table,      // res*res
    const float2* in,         // batch
    float2* out,              // batch
    uint32_t n, uint32_t res)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float scale = (float)res - 1.0f;
    float px = fmaf(scale, in[i].x, 0.5f);
    float py = fmaf(scale, in[i].y, 0.5f);
    int ix = (int)floorf(px), iy = (int)floorf(py);
    float fx = px - ix, fy = py - iy;

    float2 r = make_float2(0.f, 0.f);
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int cx = min(ix + (c & 1), (int)res - 1);
        int cy = min(iy + (c >> 1), (int)res - 1);
        float w = ((c & 1) ? fx : 1.f - fx) * ((c >> 1) ? fy : 1.f - fy);
        float2 v = table[cy * res + cx];
        r.x += w * v.x;
        r.y += w * v.y;
    }
    out[i] = r;
}
```

验证：

1. `res = 4`，table 填 `f(x, y) = x + 10*y` 这类可手算的值。
2. CPU 版用同样公式，逐元素对比，误差 < 1e-6。
3. 检查四个角 `x/y = 0 或 1`，确认不越界。

这一关过了，forward 只剩「把下标换成 hash」。

---

## 6. Demo 04~10 要点

### 04 多 level dense

- 每层 `res_l`、`table_size_l`、`offset_l` 在 host 算好。
- 用 `__constant__` 或 kernel 参数传一个小的 `LevelParams` 数组。
- 输出 `L * F` 维，按 AoS 写：`out[i * L * F + l * F + f]`。
- 验收：与 CPU 版逐元素一致。

### 05 换成 hash

```cpp
__device__ uint32_t hash2(int x, int y) {
    uint32_t h = (uint32_t)x * 1u + (uint32_t)y * 2654435761u;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
    return h;
}
```

- 每层 `size = min(res*res, T)`，8 对齐。
- `idx = (dense < size) ? dense : (hash2(cx, cy) % size)`。
- 验收：`res^2 <= T` 时与 dense 版**逐位相等**；调大 res 后图像仍连续。

### 06 half + 向量化

- 参数改 `__half`，计算用 float。
- 一次读 `__half2`（4 字节）而不是两个 `__half`。
- 注意 16B 对齐（8 个 half）。
- 验收：与 float 版输出误差 < 1e-3。

### 07 dense 反向

```cpp
float w = ...;
atomicAdd(&grad[idx * 2 + 0], w * dL_dy[i].x);
atomicAdd(&grad[idx * 2 + 1], w * dL_dy[i].y);
```

- 先用 `float` 梯度缓冲。
- 每次 backward 前 `cudaMemsetAsync(grad, 0, ...)`。
- 有限差分：`(L(p+eps) - L(p-eps)) / (2*eps)` 对 `grad[p]`。

### 08 hash 反向

- 碰撞会导致多个 cell 累加到同一地址，这是**正确行为**。
- 测量原子竞争：把 `atomicAdd` 注释掉对比时间。
- 优化：`atomicAdd(__half2*)` 一次更新两个特征（F=2 正好）。

### 09 接 tcnn

- `forward_impl` 里按 `stream` 启动 kernel，不要 `cudaDeviceSynchronize`。
- `output == nullptr` 时直接返回 context。
- padding 列写常数。
- `register_encoding<network_precision_t>("HashGridCustom", ...)`。

### 10 优化

- ncu 看 L2 命中率、访存效率。
- 按 level 分块（一个 block 处理一层），提高 L2 复用。
- batch 排序，让同一 cell 的访问集中。
- 用 shared memory 暂存当前 level 的 tile。

---

## 7. 验证方法

1. **CPU reference**：每个 demo 都写一个 `std::vector` 版本，只用于对比。
2. **有限差分**：`eps = 1e-3`，相对误差 < 1e-3 视为通过。
3. **compute-sanitizer**：`/usr/local/cuda/bin/compute-sanitizer ./demo`，查越界与竞争。
4. **nsys**：`nsys profile ./demo`，看 kernel 时间线。
5. **ncu**（需安装）：看带宽、L2、warp stall。

---

## 8. 环境与编译

```bash
export PATH=/usr/local/cuda/bin:$PATH

# 单个 demo
nvcc -arch=sm_120 -O3 -std=c++17 01_vecadd_bandwidth.cu -o 01

# 调试版（带越界检查）
nvcc -arch=sm_120 -g -G -std=c++17 03_bilinear_dense.cu -o 03_dbg
```

---

## 9. 时间安排

| 周 | 内容 |
| --- | --- |
| 第 1 周 | 01~04：索引、访存、插值、多 level |
| 第 2 周 | 05~06：hash、半精度 |
| 第 3 周 | 07~08：反向、有限差分 |
| 第 4 周 | 09~10：接入 tcnn、性能优化 |

每天 1~2 小时，每个 demo 控制在 150 行以内，带 `main()` 和 CPU 校验。

---

## 附录 A：核心公式

```
scale_l   = base * per_level_scale^l - 1      // 格子数（浮点）
res_l     = ceil(scale_l) + 1                 // 顶点数
pos       = x * scale_l + 0.5
cell      = floor(pos)
frac      = pos - cell
count_l   = min(res_l^d, T)                   // 再对齐到 8
offset_l  = sum(count_0..count_{l-1})
n_params  = offset_L * F
dense_idx = sum(cell_d * res_l^d)             // 行优先
hash_idx  = hash(cell) % count_l
```

## 附录 B：常见坑

1. `1 / width` 是整数除法 → 0（本项目踩过）。
2. `res^d` 用 32 位会溢出（3D `res=2048` 时 `8.6e9`）。
3. hash 模数必须用**每层**的 `count`，不是全局 `T`。
4. `pos = x * (res - 1)` vs `x * res` 必须全局统一。
5. `__half` 的 `atomicAdd` 慢，先 float 后优化。
6. 忘记 `cudaMemsetAsync` 清零梯度。
7. `forward` 的 `output` 可能为 `nullptr`。
8. padding 列没写常数，训练会抖。
9. 每层 offset 算错，表现为「训练能跑但效果差」。
10. 用 `std::pow` + `ceil` 时浮点误差可能多一格，构造期定死并让 kernel 复用同一函数。