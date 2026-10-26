#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vecAdd_kernel(float* a, float* b, float* c, int size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    constexpr int size = 16;
    std::vector<float> cpu_a(size, 1.f);
    std::vector<float> cpu_b(size, 1.f);
    std::vector<float> cpu_c(size, 0.f);

    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;
    cudaMalloc((void**)&gpu_a, size * sizeof(float));
    cudaMalloc((void**)&gpu_b, size * sizeof(float));
    cudaMalloc((void**)&gpu_c, size * sizeof(float));

    cudaMemcpy(gpu_a, cpu_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    vecAdd_kernel<<<1, 16>>>(gpu_a, gpu_b, gpu_c, size);
    cudaDeviceSynchronize();

    cudaMemcpy(cpu_c.data(), gpu_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (auto val : cpu_c)
    {
        std::cout << val << std::endl;
    }

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}