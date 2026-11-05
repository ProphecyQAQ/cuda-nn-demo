#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>
#include "cuda_check.h"
#include "cuda_common.h"
#include <functional>
#include <map>

void generate(std::vector<float>& a, int n)
{
    a.resize((size_t)n * n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            a[(size_t)y * n + x] = (float)(x * n + y);
}

struct CheckResult {
    long long mismatches = 0;
    int   bad_x = -1, bad_y = -1;
    float expect = 0.f, got = 0.f;
    bool  ok() const { return mismatches == 0; }
};

CheckResult verify(const std::vector<float>& in, const std::vector<float>& out,
                   int n, bool transposed = true)
{
    CheckResult r;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float expect = transposed ? in[(size_t)x * n + y]
                                            : in[(size_t)y * n + x];
            const float got    = out[(size_t)y * n + x];
            if (got != expect) {
                if (r.mismatches == 0) { r.bad_x = x; r.bad_y = y; r.expect = expect; r.got = got; }
                ++r.mismatches;
            }
        }
    }
    return r;
}

__global__ void transpose_native(float* in, float* out, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    out[x * n + y] = in[y * n + x];
}

constexpr int tile_size = 32;
__global__ void transpose_tile(float* in, float* out, int n)
{
    __shared__ float tile[tile_size + 1][tile_size + 1];

    int x = blockIdx.x * tile_size + threadIdx.x;
    int y = blockIdx.y * tile_size + threadIdx.y;

    if (x < n && y < n)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();
    
    int ox = blockIdx.y * tile_size + threadIdx.x;
    int oy = blockIdx.x * tile_size + threadIdx.y;
    if (ox < n && oy < n)
    {
        out[oy * n + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    const int n = 8192;
    const size_t nbytes = (size_t)n * (size_t)n * sizeof(float);

    std::map<std::string, std::function<TimerScope::Timing(float* gpu_in, float* gpu_out)>> funcs;

    funcs["transpose_native"] = [&](float* gpu_in, float* gpu_out){
        TimerScope::Timing t = TimerScope::timer_kernel([&]
            {
                dim3 threadsPerBlock(tile_size, tile_size);
                dim3 blocksPerGrid((n + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);
                transpose_native<<<blocksPerGrid, threadsPerBlock>>>(gpu_in, gpu_out, n);
            });
        return t;
    };

    funcs["transpose_tile"] = [&](float* gpu_in, float* gpu_out){
        TimerScope::Timing t = TimerScope::timer_kernel([&]
            {
                dim3 threadsPerBlock(tile_size, tile_size);
                dim3 blocksPerGrid((n + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);
                transpose_tile<<<blocksPerGrid, threadsPerBlock>>>(gpu_in, gpu_out, n);
            });
        return t;
    };

    for (auto& [name, func] : funcs)
    {
        std::printf("Test [%s] start\n\n", name.c_str());
        BenchReport report;

        std::vector<float> cpu_in;
        generate(cpu_in, n);

        float* gpu_in = nullptr;
        float* gpu_out = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&gpu_in, nbytes));
        CUDA_CHECK(cudaMalloc((void**)&gpu_out, nbytes));
        CUDA_CHECK(cudaMemcpy(gpu_in, cpu_in.data(), nbytes, cudaMemcpyHostToDevice));
        TimerScope::Timing t = func(gpu_in, gpu_out);
        CUDA_CHECK_LAST();

        std::vector<float> cpu_out;
        cpu_out.resize(n*n);
        CUDA_CHECK(cudaMemcpy(cpu_out.data(), gpu_out, nbytes, cudaMemcpyDeviceToHost));

        CheckResult c = verify(cpu_in, cpu_out, n, /*transposed=*/true);
        report.add(name.c_str(), t, 2.0 * nbytes, c.ok() ? 0 : c.mismatches);
        if (!c.ok())
        {
            std::printf("  第一个错: out[%d][%d] = %g, 期望 in[%d][%d] = %g\n",
                        c.bad_y, c.bad_x, c.got, c.bad_x, c.bad_y, c.expect);
        }
        else
        {
            std::printf("  验证通过\n");
            report.print(getTheoreticalPeakGBs());
        }
        CUDA_CHECK(cudaFree(gpu_in));
        CUDA_CHECK(cudaFree(gpu_out));

        std::printf("Test [%s] finish\n\n", name.c_str());
    }
}