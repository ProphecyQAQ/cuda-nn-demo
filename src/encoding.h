#pragma once

#include <cmath>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/cpp_api.h>

#include <memory>

using namespace tcnn;
template <typename T>
class HashGridEncoding : public tcnn::Encoding<T>
{
public:
    HashGridEncoding(nlohmann::json& config)
    {
        m_n_input_dims = config["n_input_dims"];
        m_n_features_per_level = config["n_features_per_level"];
        m_n_levels = config["n_levels"];
        m_hash_table_size = config["hash_table_size"];
        m_per_level_scale = config["per_level_scale"];
        m_min_resolution = config["min_resolution"];

        // calaulate other parameter level by level
        m_resolution_per_level.resize(m_n_levels);
        for (uint32_t level = 0; level < m_n_levels; level ++)
        {
            uint32_t level_res = std::ceil(m_min_resolution * std::pow(m_per_level_scale, level));
            m_resolution_per_level[level] = level_res;

            const uint64_t dense = (uint64_t)level_res * level_res;
            uint64_t count = std::min<uint64_t>(dense, m_hash_table_size);
            m_n_params += count * m_n_features_per_level;
        }
        
        // Initialize params
        m_params.resize(m_n_params);
    }

    virtual ~HashGridEncoding() = default;

    virtual std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<T>& input,
		tcnn::GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override;

    virtual void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override;

    virtual uint32_t input_width() const
    {
        return m_n_input_dims;
    }

    virtual size_t n_params() const 
    {
        return m_n_params;
    }
private:
    uint32_t m_n_input_dims;
    // size of elements in vector
    uint32_t m_n_features_per_level;
    // size of level
    uint32_t m_n_levels;

    // max size of feature vector size per level
    uint32_t m_hash_table_size = 0;
    // can be calculated
    uint64_t m_n_params = 0;

    // max resolution can be calculated by min res, level_scale and level size
    uint32_t m_min_resolution = 0, m_max_resolution = 0;
    float m_per_level_scale = 0.f;

    std::vector<uint64_t> m_resolution_per_level;

    std::vector<T> m_params;
    std::vector<T> m_grad;
};