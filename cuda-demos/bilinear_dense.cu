#include "cuda_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
constexpr int      resolution = 32;
constexpr int      n_points   = 1 << 20;   // 100 万点
constexpr int      block_size = 256;
// 表值最大 341（cy=31,cx=31），fp32 在该区间的 ulp ≈ 3.05e-5。
// 加上 CPU 是 mul+add、GPU 会被收缩成 FMA，两边必然差 1~2 ulp，所以容差不能设 1e-6。
constexpr float    abs_tol    = 1e-4f;
constexpr unsigned rng_seed   = 12345u;    // 固定种子，保证误差可复现

// ---------------------------------------------------------------------------
// 特征表：填一个线性函数 f(cx,cy) = (cx + 10cy, 3cx - 5cy)。
// 线性函数能被双线性插值精确复原，所以可以拿解析式当第二重 reference。
// ---------------------------------------------------------------------------
std::vector<float2> make_table(int res)
{
    std::vector<float2> table((size_t)res * res);

    for (int cy = 0; cy < res; ++cy) {
        for (int cx = 0; cx < res; ++cx) {
            float2 v;
            v.x = (float)cx + 10.0f * (float)cy;
            v.y = 3.0f * (float)cx - 5.0f * (float)cy;
            table[(size_t)cy * res + cx] = v;
        }
    }
    return table;
}

// ---------------------------------------------------------------------------
// 输入：固定种子 + [0,1) 均匀分布（不含 1.0，所以 px 最小 0.5，永远非负）
// ---------------------------------------------------------------------------
std::vector<float> initialize_input(int size, unsigned seed)
{
    std::vector<float> inputs((size_t)size * 2);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    for (int ind = 0; ind < size; ++ind) {
        inputs[ind * 2 + 0] = dis(gen);
        inputs[ind * 2 + 1] = dis(gen);
    }
    return inputs;
}

// ---------------------------------------------------------------------------
// CPU / GPU 共用的数学
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

__host__ __device__ __forceinline__
float2 bilinear_interpolation(const float2& feat_00, const float2& feat_01,
                              const float2& feat_10, const float2& feat_11,
                              float dx, float dy)
{
    float2 res{0.f, 0.f};

    res.x = feat_00.x * (1.f - dx) * (1.f - dy) + feat_10.x * dx * (1.f - dy)
          + feat_01.x * (1.f - dx) * dy       + feat_11.x * dx * dy;
    res.y = feat_00.y * (1.f - dx) * (1.f - dy) + feat_10.y * dx * (1.f - dy)
          + feat_01.y * (1.f - dx) * dy       + feat_11.y * dx * dy;

    return res;
}

// 网格查找：CPU / GPU 共用。
// x1/y1 永远是 x0+1 / y0+1 再夹上界，不用 ceil —— 采样点踩在顶点上时
// 由权重 fx=0 自动屏蔽右/上邻居，保证单元格不退化。
__host__ __device__ __forceinline__
float2 bilinear_lookup(const float2* __restrict__ table, float px, float py, int res)
{
    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);

    float fx = px - (float)x0;
    float fy = py - (float)y0;

    int x1 = imin(x0 + 1, res - 1);
    int y1 = imin(y0 + 1, res - 1);

    return bilinear_interpolation(table[y0 * res + x0], table[y1 * res + x0],
                                  table[y0 * res + x1], table[y1 * res + x1], fx, fy);
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void grid_cpu(const float2* table, const float* in, float* out, int n_points, int res)
{
    for (int ind = 0; ind < n_points; ++ind) {
        float x = std::fmaf(in[ind * 2 + 0], (float)(res - 1), 0.5f);
        float y = std::fmaf(in[ind * 2 + 1], (float)(res - 1), 0.5f);

        float2 res_feat = bilinear_lookup(table, x, y, res);
        out[ind * 2 + 0] = res_feat.x;
        out[ind * 2 + 1] = res_feat.y;
    }
}

__global__ void bilinear_dense_kernel(const float2* __restrict__ table,
                                      const float2* __restrict__ in,
                                      float2* __restrict__ out,
                                      int n_points, int res)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    const float scale = (float)(res - 1);
    float px = fmaf(scale, in[idx].x, 0.5f);
    float py = fmaf(scale, in[idx].y, 0.5f);

    out[idx] = bilinear_lookup(table, px, py, res);
}

float2 expected_linear(float px, float py, int res)
{
    const float cl = (float)(res - 1);
    float qx = std::min(px, cl);
    float qy = std::min(py, cl);
    return float2{ qx + 10.f * qy, 3.f * qx - 5.f * qy };
}

int main()
{
    std::printf("resolution = %d, n_points = %d (%.2f M)\n",
                resolution, n_points, n_points / 1e6);

    std::vector<float2> table  = make_table(resolution);
    std::vector<float>  inputs = initialize_input(n_points, rng_seed);

    std::vector<float2> out_cpu(n_points);
    std::vector<float2> out_gpu(n_points);

    BenchReport report;

    const float probes[][2] = {
        { 0.5f,  0.5f  },                                       // 首格中心
        { 1.0f,  1.0f  },                                       // 踩顶点 (1,1)
        { 8.0f, 16.0f  },                                       // 踩顶点 (8,16)
        { 8.5f, 16.5f  },                                       // 格心
        { (float)(resolution - 1), 3.25f },                     // 右边界上的顶点
        { (float)resolution - 0.5f, (float)resolution - 0.5f }, // 右上角，触发 clamp
    };

    int   probe_failed  = 0;
    float probe_max_err = 0.f;

    std::printf("\n[check] bilinear_lookup vs 解析式（表是线性函数）\n");
    for (const auto& p : probes) {
        float px = p[0], py = p[1];
        float2 got  = bilinear_lookup(table.data(), px, py, resolution);
        float2 want = expected_linear(px, py, resolution);
        float  e    = std::max(std::fabs(got.x - want.x), std::fabs(got.y - want.y));

        probe_max_err = std::max(probe_max_err, e);
        if (e > abs_tol) ++probe_failed;

        std::printf("  px=%7.3f py=%7.3f  got(%9.4f, %9.4f)  want(%9.4f, %9.4f)  err=%.3e%s\n",
                    px, py, got.x, got.y, want.x, want.y, e,
                    e > abs_tol ? "   <-- FAIL" : "");
    }

    // =======================================================================
    // 1. 显存分配 + H2D
    // =======================================================================
    float2* gpu_table   = nullptr;
    float2* gpu_inputs  = nullptr;
    float2* gpu_outputs = nullptr;

    const size_t bytes_table = (size_t)resolution * resolution * sizeof(float2);
    const size_t bytes_io    = (size_t)n_points * sizeof(float2);

    CUDA_CHECK(cudaMalloc((void**)&gpu_table,   bytes_table));
    CUDA_CHECK(cudaMalloc((void**)&gpu_inputs,  bytes_io));
    CUDA_CHECK(cudaMalloc((void**)&gpu_outputs, bytes_io));

    CUDA_CHECK(cudaMemcpy(gpu_table,  table.data(),  bytes_table, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_inputs, inputs.data(), bytes_io,    cudaMemcpyHostToDevice));

    // =======================================================================
    // 2. GPU 计时（只包 kernel，不含 memcpy）
    // =======================================================================
    const int grid = (n_points + block_size - 1) / block_size;

    TimerScope::Timing timing = TimerScope::timer_kernel([&] {
        bilinear_dense_kernel<<<grid, block_size>>>(gpu_table, gpu_inputs, gpu_outputs,
                                                    n_points, resolution);
    });
    CUDA_CHECK(cudaGetLastError());   // 捕获 launch 配置错误（越界、资源不足等）

    // =======================================================================
    // 3. D2H + CPU reference 计时
    // =======================================================================
    CUDA_CHECK(cudaMemcpy(out_gpu.data(), gpu_outputs, bytes_io, cudaMemcpyDeviceToHost));

    constexpr int cpu_reps = 3;
    double cpu_best = 1e30, cpu_sum = 0.0;
    for (int r = 0; r < cpu_reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        grid_cpu(table.data(), inputs.data(), reinterpret_cast<float*>(out_cpu.data()),
                 n_points, resolution);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_best = std::min(cpu_best, ms);
        cpu_sum += ms;
    }
    TimerScope::Timing timing_cpu{ cpu_best, cpu_sum / cpu_reps };

    // =======================================================================
    // 4. 逐元素对比，超出容差算 mismatch
    // =======================================================================
    long long mismatches = 0;
    float     max_err    = 0.f;
    int       worst      = -1;

    for (int i = 0; i < n_points; ++i) {
        float e = std::max(std::fabs(out_cpu[i].x - out_gpu[i].x),
                           std::fabs(out_cpu[i].y - out_gpu[i].y));
        if (e > max_err) { max_err = e; worst = i; }
        if (e > abs_tol) ++mismatches;
    }

    // =======================================================================
    // 5. 报告
    // =======================================================================
    // 读写各一次 float2。注意这张表只有 8 KB、完全常驻 L1，
    // 所以下面的 GB/s 不代表显存带宽瓶颈，看 Mpts/s 更有意义。
    const double bytes_moved = 2.0 * (double)n_points * sizeof(float2);

    report.add("gpu bilinear", timing, bytes_moved, mismatches);
    report.add("cpu ref (1 thread)", timing_cpu, bytes_moved, 0);
    report.print(getTheoreticalPeakGBs());

    const double mpts_per_s = n_points / (timing.best_ms * 1e3);
    const double speedup    = timing_cpu.best_ms / timing.best_ms;

    std::printf("\nmax_abs_err = %.3e   (worst point #%d, tol %.0e)\n", max_err, worst, abs_tol);
    std::printf("probe check : %s   (max err %.3e)\n",
                probe_failed == 0 ? "PASS" : "FAIL", probe_max_err);
    std::printf("throughput  : %.2f Mpts/s\n", mpts_per_s);
    std::printf("speedup     : cpu / gpu = %.1fx\n", speedup);
    std::printf("NOTE: 8 KB 特征表全程命中 L1，此处的 GB/s 不是带宽指标。\n");

    // =======================================================================
    // 6. 收尾
    // =======================================================================
    CUDA_CHECK(cudaFree(gpu_outputs));
    CUDA_CHECK(cudaFree(gpu_inputs));
    CUDA_CHECK(cudaFree(gpu_table));

    const bool ok = (mismatches == 0) && (probe_failed == 0);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
