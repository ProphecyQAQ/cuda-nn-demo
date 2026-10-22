#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <tiny-cuda-nn/common_host.h>

#include <memory>
#include <cmath>
#include <fmt/core.h>

__device__ __host__ inline float grid_scale(uint32_t level, float log2_per_level_scale, uint32_t base_resolution)
{
    return std::exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
}

__device__ __host__ inline uint32_t grid_resolution(float scale) 
{
    return (uint32_t)ceilf(scale) + 1;
}

template <typename T>
class HashGridEncoding : public tcnn::Encoding<T>
{
public:
    HashGridEncoding(const nlohmann::json& config)
    {
        m_n_input_dims = config["n_input_dims"];
        m_n_features_per_level = config["n_features_per_level"];
        m_n_levels = config["n_levels"];
        m_max_hash_table_size = config["max_hash_table_size"];
        m_per_level_scale = config["per_level_scale"];
        m_min_resolution = config["min_resolution"];

        // calaulate other parameter level by level
        m_resolution_per_level.resize(m_n_levels);
        m_level_params.resize(m_n_levels);
        m_log2_per_level_scale = std::log2(m_per_level_scale);
        for (uint32_t level = 0; level < m_n_levels; level ++)
        {
            uint32_t level_res =  grid_resolution(grid_scale(level, m_log2_per_level_scale, m_min_resolution));
            m_resolution_per_level[level] = level_res;

            const uint64_t dense = (uint64_t)level_res * level_res;
            m_level_params[level].hash_table_size = std::min<uint64_t>(dense, m_max_hash_table_size);
            m_level_params[level].padded_hash_table_size = tcnn::next_multiple<uint64_t>(m_level_params[level].hash_table_size, 8u);
            m_n_params += m_level_params[level].padded_hash_table_size * m_n_features_per_level;
            
            fmt::print("Level {}, resolution: {}, hash table size: {}, params_size: {}", level, level_res, m_level_params[level].padded_hash_table_size, m_level_params[level].padded_hash_table_size * m_n_features_per_level);
        }

        m_n_padded_output_dims = output_width();
    }

    virtual ~HashGridEncoding() = default;

#pragma region params

    virtual size_t n_params() const 
    {
        return m_n_params;
    }

    virtual void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale) override 
    {
        tcnn::generate_random_uniform(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
    }

    virtual void set_params_impl(T* params, T* inference_params, T* gradients) override 
    {   
        uint64_t offset = 0;
        for (uint32_t level = 0; level < m_n_levels; level ++)
        {
            m_level_params[level].params           = params + offset;   
            m_level_params[level].gradients        = gradients + offset;   
            m_level_params[level].inference_params = inference_params + offset;   

            offset += m_level_params[level].padded_hash_table_size * m_n_features_per_level;
        }
    } 

#pragma endregion params

    virtual std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override
    {
        
    }

    virtual void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::GradientMode param_gradients_mode = tcnn::GradientMode::Overwrite
	) override;

    virtual uint32_t input_width() const override
    {
        return m_n_input_dims;
    }

    virtual uint32_t required_input_alignment() const override { return 1; }

    virtual uint32_t output_width() const override
    {
        return m_n_features_per_level * m_n_levels;
    }

    uint32_t padded_output_width() const override 
    {
        return m_n_padded_output_dims;
    }

    void set_padded_output_width(uint32_t w) override 
    {
        CHECK_THROW(w >= output_width());
        m_n_padded_output_dims = w;
    }

    virtual uint32_t required_output_alignment() const override { return 1; }

    virtual nlohmann::json hyperparams() const override {
    return {{"otype", "HashGridCustom"},
            {"n_levels", m_n_levels},
            {"n_features_per_level", m_n_features_per_level},
            {"max_hash_table_size", m_max_hash_table_size},
            {"min_resolution", m_min_resolution},
            {"per_level_scale", m_per_level_scale}};
    }

    // In tcnn, input fllow (width * batch)
    tcnn::MatrixLayout preferred_output_layout() const override { return tcnn::MatrixLayout::ColumnMajor; }

private:
    uint32_t m_n_input_dims;
    uint32_t m_n_padded_output_dims;
    // size of elements in vector
    uint32_t m_n_features_per_level;
    // size of level
    uint32_t m_n_levels;

    // max size of feature vector size per level
    uint32_t m_max_hash_table_size = 0;
    // can be calculated
    uint64_t m_n_params = 0;

    // max resolution can be calculated by min res, level_scale and level size
    uint32_t m_min_resolution = 0, m_max_resolution = 0;
    float m_per_level_scale = 0.f;
    float m_log2_per_level_scale = 0.f;

    std::vector<uint64_t> m_resolution_per_level;

    struct LevelParams
    {
        LevelParams() = default;

        uint64_t hash_table_size;
        // allige 8 byte
        uint64_t padded_hash_table_size;

        T* params = nullptr;
        T* inference_params = nullptr;
        T* gradients = nullptr;
    };
    std::vector<LevelParams> m_level_params;
};