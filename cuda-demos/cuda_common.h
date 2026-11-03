#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.h"

// ---------------------------------------------------------------------------
// 计时：warmup 若干次，然后每轮单独计时，同时给 best 与 avg
// ---------------------------------------------------------------------------
class TimerScope
{
public:
    struct Timing { double best_ms, avg_ms; };

    template<typename T>
    static Timing timer_kernel(T kernel, int warmup = 3, int reps = 10)
    {
        for (int i = 0; i < warmup; ++i) kernel();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double best = 1e30, sum = 0.0;
        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaEventRecord(start));
            kernel();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            best = std::min<double>(best, ms);
            sum += ms;
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return {best, sum / reps};
    }
};

// ---------------------------------------------------------------------------
// 理论峰值带宽：2(双沿) * 显存频率 * 位宽 / 8，单位 GB/s（十进制）
// CUDA 13 起 cudaDeviceProp 不再有 memoryClockRate，只能用 device attribute
// ---------------------------------------------------------------------------
inline double getTheoreticalPeakGBs()
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int mem_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));

    return 2.0 * mem_khz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
}

// ---------------------------------------------------------------------------
// 结果表：一行一个 kernel，最后统一打印
//   report.add("名字", 计时结果, 搬运字节数, 校验错误数);
//   校验错误数传 -1 表示这一行不做正确性校验（比如 memcpy 基线）
// ---------------------------------------------------------------------------
class BenchReport
{
public:
    void add(const char* name, TimerScope::Timing t, double bytes, long long mismatches = -1)
    {
        Row row;
        row.name       = name;
        row.best_ms    = t.best_ms;
        row.avg_ms     = t.avg_ms;
        row.bytes      = bytes;
        row.mismatches = mismatches;
        m_rows.push_back(row);
    }

    void print(double peak_gbs) const
    {
        std::printf("\n%-18s %10s %10s %10s %8s %8s %11s\n",
                    "variant", "best ms", "avg ms", "GB/s", "%peak", "jitter", "mismatches");
        std::printf("%s\n",
                    "---------------------------------------------------------------------------------");

        for (const Row& r : m_rows) {
            const double gbs = r.bytes / (r.best_ms * 1e-3) / 1e9;
            const double jit = 100.0 * (r.avg_ms - r.best_ms) / r.best_ms;
            const double pct = 100.0 * gbs / peak_gbs;

            if (r.mismatches < 0) {
                std::printf("%-18s %10.4f %10.4f %10.1f %7.1f%% %7.1f%% %11s\n",
                            r.name.c_str(), r.best_ms, r.avg_ms, gbs, pct, jit, "-");
            } else {
                std::printf("%-18s %10.4f %10.4f %10.1f %7.1f%% %7.1f%% %11lld\n",
                            r.name.c_str(), r.best_ms, r.avg_ms, gbs, pct, jit, r.mismatches);
            }
        }

        std::printf("%s\n",
                    "---------------------------------------------------------------------------------");
        std::printf("theoretical peak %.1f GB/s   (jitter = (avg-best)/best，越大说明抖动越厉害)\n",
                    peak_gbs);
    }

private:
    struct Row {
        std::string name;
        double      best_ms = 0.0;
        double      avg_ms  = 0.0;
        double      bytes   = 0.0;
        long long   mismatches = -1;
    };

    std::vector<Row> m_rows;
};

int getTotalSMNum()
{
    int num = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num, cudaDevAttrMultiProcessorCount, 0));   
    return num;
}

int getMaxThreadsNumPerSM()
{
    int num = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num, cudaDevAttrMaxThreadsPerMultiProcessor, 0));   
    return num;
}

// 全卡同时能驻留多少线程
int getMaxThreadsNum()
{
    return getTotalSMNum() * getMaxThreadsNumPerSM();
}