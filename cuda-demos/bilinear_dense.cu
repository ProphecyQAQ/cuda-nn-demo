#include "cuda_common.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

constexpr int resolution = 32;

std::vector<float2> make_table(int res)
{
    std::vector<float2> table(res * res);

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

std::vector<float> initialize_input(int size)
{
    std::vector<float> inputs(size * 2);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis;

    for (int ind = 0; ind < size; ind ++)
    {
        inputs[ind * 2] = dis(gen);
        inputs[ind * 2 + 1] = dis(gen);
    }

    return inputs;
}

float2 bilinear_interpolation(const float2& feat_00, const float2& feat_01, const float2& feat_10, const float2& feat_11, float dx, float dy)
{
    float2 res{0.f, 0.f};

    res.x = feat_00.x * (1.f - dx) * (1.f - dy) + feat_10.x * dx * (1.f - dy) + feat_01.x * (1 - dx) * dy + feat_11.x * dx * dy; 
    res.y = feat_00.y * (1.f - dx) * (1.f - dy) + feat_10.y * dx * (1.f - dy) + feat_01.y * (1 - dx) * dy + feat_11.y * dx * dy; 

    return res;
}

void print_results(
    const std::vector<float>& inputs,
    const std::vector<float2>& outputs,
    int n_points)
{
    for (int i = 0; i < n_points; ++i)
    {
        std::cout << "Point " << i << ":\n";

        std::cout << "  Input : ("
                  << inputs[i * 2] << ", "
                  << inputs[i * 2 + 1] << ")\n";

        std::cout << "  Output: ("
                  << outputs[i].x << ", "
                  << outputs[i].y << ")\n";
    }
}

void grid_cpu(float2* table, float* in, float* out, int n_points, int res)
{
    for (int ind = 0; ind < n_points; ind ++)
    {
        float x = in[ind * 2] * (res - 1) + 0.5f;
        float y = in[ind * 2 + 1] * (res - 1) + 0.5f;
        
        int dw_x = static_cast<int>(std::floor(x));
        int dw_y = static_cast<int>(std::floor(y));

        float dx = x - dw_x;
        float dy = y - dw_y;

        int up_x = std::min(dw_x + 1, res - 1);
        int up_y = std::min(dw_y + 1, res - 1);

        float2 res_feat = bilinear_interpolation(table[dw_y * res + dw_x], table[up_y * res + dw_x], table[dw_y * res + up_x], table[up_y * res + up_x], dx, dy);
        out[ind * 2] = res_feat.x;
        out[ind * 2 + 1] = res_feat.y;
    }
}

int main()
{
    std::vector<float2> table = make_table(resolution);

    int n_points = 10;
    std::vector<float> inputs = initialize_input(n_points);
    std::vector<float2> outputs(10);

    grid_cpu(table.data(), inputs.data(), reinterpret_cast<float*>(outputs.data()), n_points, resolution);

    print_results(inputs, outputs, n_points);

    return 0;
}