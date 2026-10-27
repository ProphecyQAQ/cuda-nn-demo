#pragma once

// ---------------------------------------------------------------------------
// CUDA 错误检查
//
//   CUDA_CHECK(call)   检查任意返回 cudaError_t 的调用（同步错误，当场返回）
//   CUDA_CHECK_LAST()  kernel<<<...>>> 之后立刻调，抓“启动错误”
//   CUDA_CHECK_SYNC()  等 kernel 跑完并检查，抓“执行期错误”（越界、非法地址）
//
// 失败时打印 文件:行 / 调用表达式 / 错误码+名字+描述，然后 exit(EXIT_FAILURE)。
//
// 为什么 kernel 要单独检查：
//   kernel<<<...>>> 本身不返回错误码。启动错误（线程数超限、shared mem 超限）
//   要靠 cudaGetLastError() 拿；执行期错误（越界写、野指针）还要等同步才暴露。
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr,                                                  \
                "\n[CUDA ERROR]\n"                                                \
                "  where : %s:%d\n"                                               \
                "  call  : %s\n"                                                  \
                "  code  : %d (%s)\n"                                             \
                "  what  : %s\n",                                                 \
                __FILE__, __LINE__, #call, (int)err_,                             \
                cudaGetErrorName(err_), cudaGetErrorString(err_));                \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define CUDA_CHECK_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
