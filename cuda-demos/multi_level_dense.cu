#include "cuda_check.h"
#include "cuda_common.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>

struct LevelTables
{
    std::vector<float2> table;
    std::vector<int> resolution;
    std::vector<int> table_size;
    std::vector<int> padded_table_size;
    std::vector<int> offset;
};

__host__ __device__ __forceinline__
uint32_t hash_table(int x, int y, int mod)
{
    return ((uint32_t)x*1u ^ (uint32_t)y*2654435761u) % mod;
}

template <typename T>
__host__ __device__ __forceinline__ T next_multiple(T val, T divisor) {
	return ((val + divisor - 1) / divisor) * divisor;
}

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

__host__ __device__ __forceinline__
float2 bilinear_lookup(const float2* __restrict__ table, float px, float py, int res, int table_size)
{
    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);

    float fx = px - (float)x0;
    float fy = py - (float)y0;

    int x1 = imin(x0 + 1, res - 1);
    int y1 = imin(y0 + 1, res - 1);

    bool hashed = res * res > table_size;
    int feat_00_idx = hashed ? hash_table(x0, y0, table_size) : y0 * res + x0;
    int feat_01_idx = hashed ? hash_table(x0, y1, table_size) : y1 * res + x0;
    int feat_10_idx = hashed ? hash_table(x1, y0, table_size) : y0 * res + x1;
    int feat_11_idx = hashed ? hash_table(x1, y1, table_size) : y1 * res + x1;

    return bilinear_interpolation(table[feat_00_idx], table[feat_01_idx],
                                  table[feat_10_idx], table[feat_11_idx], fx, fy);
}

float grid_scale(int level, float per_level_scale, int base_resolution)
{
    static float log2_per_level_scale = std::log2(per_level_scale);
    return base_resolution * std::exp2(level * log2_per_level_scale) - 1.f;
}

int grid_resolution(float grid_scale)
{
    return static_cast<int>(std::ceil(grid_scale) + 1.f);
}

std::vector<float2> initialize_input(int size, unsigned seed)
{
    std::vector<float2> inputs((size_t)size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    for (int ind = 0; ind < size; ++ind) {
        inputs[ind].x = dis(gen);
        inputs[ind].y = dis(gen);
    }
    return inputs;
}

LevelTables build_level(int n_level, float per_level_scale, int n_feat_per_level, int base_resolution, int max_table_size_per_level, bool random_table = false)
{
    LevelTables level_tables;

    int curr_table_size = 0;
    level_tables.resolution.resize(n_level);
    level_tables.offset.resize(n_level);
    level_tables.padded_table_size.resize(n_level);
    level_tables.table_size.resize(n_level);
    for (int level = 0; level < n_level; level ++)
    {
        int level_resolution = grid_resolution(grid_scale(level, per_level_scale, base_resolution));
        level_tables.resolution[level] = level_resolution;

        level_tables.offset[level] = curr_table_size;
        int table_size = std::min(level_resolution * level_resolution, max_table_size_per_level);
        int padded_table_size = static_cast<int>(next_multiple<uint64_t>((uint64_t)table_size, 8u));
        level_tables.table_size[level] = table_size;
        level_tables.padded_table_size[level] = padded_table_size;
        curr_table_size += padded_table_size;

        std::printf("Level %d: res=%4d  size=%7d  padded=%7d  %s\n",
            level, level_resolution, table_size, padded_table_size,
            (level_resolution * level_resolution <= table_size) ? "dense" : "hash");
    }

    level_tables.table.assign(curr_table_size, make_float2(0.f, 0.f));

    std::mt19937 gen(2024u);
    std::uniform_real_distribution<float> dis(-1.f, 1.f);

    for (int level = 0; level < n_level; ++level)
    {
        float2* __restrict__ base = level_tables.table.data() + level_tables.offset[level];
        const int  res    = level_tables.resolution[level];
        const int  table_size = level_tables.table_size[level];
        const bool dense  = ((size_t)res * (size_t)res <= (size_t)table_size);

        for (int idx = 0; idx < table_size; ++idx)
        {
            if (random_table) {
                base[idx] = make_float2(dis(gen), dis(gen));
            } else if (dense) {
                const int cx = idx % res;
                const int cy = idx / res;
                base[idx] = make_float2((float)cx + 10.f * (float)cy,
                                        3.f * (float)cx - 5.f * (float)cy);
            } else {
                base[idx] = make_float2((float)idx + 1.f, (float)(idx % 7) - 3.f);
            }
        }
    }

    std::printf("total vertices = %d (%.2f MB)\n", curr_table_size, curr_table_size * 8.0 / 1e6);
    return level_tables;
}

__global__ void multi_level_dense_kernel(
    const float2* __restrict__ in, 
    float* out, 
    const float2* __restrict__ table, 
    const int* __restrict__ resolution_per_level, 
    const int* __restrict__ offset_per_level,
    const int* __restrict__ table_size,
    int max_table_size_per_level, int n_input_size, int n_level, int n_feat_per_level)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n_out_dim_per_point = n_level * n_feat_per_level;
    if (idx >= n_input_size) return;

    for (int level = 0; level < n_level; level ++)
    {
        float scale = (float)(resolution_per_level[level] - 1);

        float px = fmaf(in[idx].x, scale, 0.5f);
        float py = fmaf(in[idx].y, scale, 0.5f);
        float2 lookup_feat_level = bilinear_lookup(table + offset_per_level[level], px, py, resolution_per_level[level], table_size[level]);

        out[n_out_dim_per_point * idx + level * n_feat_per_level] = lookup_feat_level.x;
        out[n_out_dim_per_point * idx + level * n_feat_per_level + 1] = lookup_feat_level.y;
    }
}

// CPU Validation
void multilevel_cpu(const LevelTables& level_table, const std::vector<float2>& in,
                    std::vector<float>& out, int n_level, int n_feat)
{
    const int stride = n_level * n_feat;
    for (size_t i = 0; i < in.size(); ++i) 
    {
        for (int level = 0; level < n_level; ++level) 
        {
            const int   res = level_table.resolution[level];
            const float s   = (float)(res - 1);
            const float2 v  = bilinear_lookup(level_table.table.data() + level_table.offset[level],
                                              std::fmaf(in[i].x, s, 0.5f),
                                              std::fmaf(in[i].y, s, 0.5f), res, level_table.table_size[level]);
            out[i * stride + level * n_feat + 0] = v.x;
            out[i * stride + level * n_feat + 1] = v.y;
        }
    }
}

int main()
{
    constexpr int n_level = 7;
    constexpr int base_resolution = 16;
    constexpr int n_feat_per_level = 2;
    constexpr float per_level_scale = 2;
    constexpr int n_input_size = 1 << 20;
    constexpr int max_table_size_per_level = 256 * 256;

    BenchReport report;

    LevelTables level_table = build_level(n_level, per_level_scale, n_feat_per_level, base_resolution, max_table_size_per_level, true);

    std::vector<float2> cpu_in = initialize_input(n_input_size, 123);
    std::vector<float> cpu_out(n_feat_per_level * n_level * n_input_size);

    float2* gpu_in = nullptr;
    float* gpu_out = nullptr;
    float2* gpu_table = nullptr;
    int* gpu_resolution_per_level = nullptr;
    int* gpu_offset_per_level = nullptr;
    int* gpu_table_size = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&gpu_in, sizeof(float2) * cpu_in.size()));
    CUDA_CHECK(cudaMalloc((void**)&gpu_out, sizeof(float) * cpu_out.size()));
    CUDA_CHECK(cudaMalloc((void**)&gpu_table, sizeof(float2) * level_table.table.size()));
    CUDA_CHECK(cudaMalloc((void**)&gpu_resolution_per_level, sizeof(int) * level_table.resolution.size()));
    CUDA_CHECK(cudaMalloc((void**)&gpu_offset_per_level, sizeof(int) * level_table.offset.size()));
    CUDA_CHECK(cudaMalloc((void**)&gpu_table_size, sizeof(int) * level_table.table_size.size()));

    CUDA_CHECK(cudaMemcpy(gpu_in, cpu_in.data(), sizeof(float2) * cpu_in.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_table, level_table.table.data(), sizeof(float2) * level_table.table.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_resolution_per_level, level_table.resolution.data(), sizeof(int) * level_table.resolution.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_offset_per_level, level_table.offset.data(), sizeof(int) * level_table.offset.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_table_size, level_table.table_size.data(), sizeof(int) * level_table.table_size.size(), cudaMemcpyHostToDevice));

    TimerScope::Timing t = TimerScope::timer_kernel([&]
    {
        int n_threads = 256;
        dim3 blocks((n_input_size + n_threads - 1)/n_threads);
        dim3 threads(n_threads);
        multi_level_dense_kernel<<<blocks, threads>>>(
            gpu_in, 
            gpu_out, 
            gpu_table, 
            gpu_resolution_per_level, 
            gpu_offset_per_level,
            gpu_table_size,
            max_table_size_per_level,
            n_input_size, 
            n_level, 
            n_feat_per_level);
    });
    CUDA_CHECK_SYNC();

    CUDA_CHECK(cudaMemcpy(cpu_out.data(), gpu_out, sizeof(float) * cpu_out.size(), cudaMemcpyDeviceToHost));

    std::vector<float> ref(cpu_out.size());
    multilevel_cpu(level_table, cpu_in, ref, n_level, n_feat_per_level);

    constexpr float abs_tol = 1e-4f;
    constexpr float rel_tol = 1e-6f;

    float     max_abs = 0.f, max_rel = 0.f;
    long long bad     = 0;
    int       worst   = 0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const float e   = std::fabs(ref[i] - cpu_out[i]);
        const float tol = abs_tol + rel_tol * std::fabs(ref[i]);
        const float rel = e / std::max(1e-6f, std::fabs(ref[i]));

        if (e > max_abs) { max_abs = e; worst = (int)i; }
        if (rel > max_rel) max_rel = rel;
        if (e > tol) ++bad;
    }

    const int stride = n_level * n_feat_per_level;
    std::printf("max_abs_err = %.3e   max_rel_err = %.3e\n", max_abs, max_rel);
    std::printf("worst       : point %d, dim %d\n", worst / stride, worst % stride);
    std::printf("              gpu=%.9g  cpu=%.9g\n", (double)cpu_out[worst], (double)ref[worst]);
    std::printf("mismatches  : %lld / %zu\n", bad, ref.size());
    std::printf("RESULT: %s\n", bad == 0 ? "PASS" : "FAIL");

    const double bytes = (double)cpu_out.size() * sizeof(float) + (double)n_input_size * sizeof(float2) + (double)(level_table.offset.back() + level_table.padded_table_size.back()) * sizeof(float2);
    report.add("gpu multilevel", t, bytes, bad);
    report.print(getTheoreticalPeakGBs());

    CUDA_CHECK(cudaFree(gpu_in));
    CUDA_CHECK(cudaFree(gpu_out));
    CUDA_CHECK(cudaFree(gpu_table));
    CUDA_CHECK(cudaFree(gpu_resolution_per_level));
    CUDA_CHECK(cudaFree(gpu_offset_per_level));
    CUDA_CHECK(cudaFree(gpu_table_size));

    return 0;
} 