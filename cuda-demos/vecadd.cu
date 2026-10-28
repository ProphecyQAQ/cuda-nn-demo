#include <iostream>
#include <vector>
#include <cstdint>
#include "cuda_check.h"

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

int main()
{
    constexpr int size = 512;
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

    // one by one
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + 256 - 1)/256);
    vecAdd_onebyone<<<blocksPerGrid, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, size);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();

    CUDA_CHECK(cudaMemcpy(cpu_c.data(), gpu_c, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (auto val : cpu_c)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    const int n = 32;
    dim3 threadsPerBlock1(n);
    dim3 blocksPerGrid1(2);
    vecAdd_stride<<<blocksPerGrid1, threadsPerBlock1>>>(gpu_a, gpu_b, gpu_c, size);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();

    CUDA_CHECK(cudaMemcpy(cpu_c.data(), gpu_c, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (auto val : cpu_c)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(gpu_a));
    CUDA_CHECK(cudaFree(gpu_b));
    CUDA_CHECK(cudaFree(gpu_c));

    return 0;
}
