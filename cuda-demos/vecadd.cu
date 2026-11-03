#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>
#include "cuda_common.h"

__global__ void vecAdd_onebyone(float* a, float* b, float* c, int size)
{
    uint32_t g_blockIdx = gridDim.x * blockIdx.y + blockIdx.x;
    uint32_t idx = g_blockIdx * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vecAdd_stride(float* a, float* b, float* c, int size)
{
    uint32_t t_total = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    uint32_t g_blockIdx = gridDim.x * blockIdx.y + blockIdx.x;
    uint32_t start_idx = (g_blockIdx * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x);

    for (size_t idx = start_idx; idx < size; idx += t_total)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void initialize(std::vector<float>& vec)
{
    for (size_t i = 0; i < vec.size(); i ++)
    {
        vec[i] = static_cast<float>(i);
    }
}

// 返回不匹配的元素个数，0 表示通过
uint64_t validation(const std::vector<float>& a, const std::vector<float>& b,
                    const std::vector<float>& c)
{
    uint64_t cnt = 0;
    for (size_t id = 0; id < a.size(); id ++)
    {
        if (c[id] != a[id] + b[id])
        {
            cnt ++;
        }
    }
    return cnt;
}

int main()
{
    constexpr int size = 64u << 20;
    std::vector<float> cpu_a(size, 1.f);
    std::vector<float> cpu_b(size, 0.f);
    std::vector<float> cpu_c(size, 0.f);
    initialize(cpu_a);

    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&gpu_a, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_b, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_c, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gpu_a, cpu_a.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_b, cpu_b.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    const double bytes3 = 3.0 * size * sizeof(float);     // 读 a + 读 b + 写 c
    const double bytes2 = 2.0 * size * sizeof(float);     // 只读 + 写（memcpy 基线）
    const double peak   = getTheoreticalPeakGBs();

    BenchReport report;

    CUDA_CHECK(cudaMemset(gpu_c, 0xFF, sizeof(float) * size));   // 哨兵：没被写的元素会保持 NaN

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + 256 - 1)/256);

    TimerScope::Timing t1 = TimerScope::timer_kernel([&]{
        vecAdd_onebyone<<<blocksPerGrid, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, size);
    });
    CUDA_CHECK(cudaMemcpy(cpu_c.data(), gpu_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    report.add("one-by-one", t1, bytes3, (long long)validation(cpu_a, cpu_b, cpu_c));

    CUDA_CHECK(cudaMemset(gpu_c, 0xFF, sizeof(float) * size));

    const int threadsNum = 256;
    const int needed = (size + threadsNum - 1) / threadsNum;
    dim3 threadsPerBlock1(threadsNum);
    dim3 blocksPerGrid1(std::max(1, std::min(needed, getMaxThreadsNum()/threadsNum)));

    TimerScope::Timing t2 = TimerScope::timer_kernel([&]{
        vecAdd_stride<<<blocksPerGrid1, threadsPerBlock1>>>(gpu_a, gpu_b, gpu_c, size);
    });
    CUDA_CHECK(cudaMemcpy(cpu_c.data(), gpu_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    report.add("stride (1 wave)", t2, bytes3, (long long)validation(cpu_a, cpu_b, cpu_c));

    TimerScope::Timing t3 = TimerScope::timer_kernel([&]{
        CUDA_CHECK(cudaMemcpyAsync(gpu_c, gpu_a, size * sizeof(float), cudaMemcpyDeviceToDevice));
    });
    report.add("memcpy D2D", t3, bytes2, -1);

    report.print(peak);

    CUDA_CHECK(cudaFree(gpu_a));
    CUDA_CHECK(cudaFree(gpu_b));
    CUDA_CHECK(cudaFree(gpu_c));

    return 0;
}
