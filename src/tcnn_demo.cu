// tiny-cuda-nn demo -- learn a 2D image with a hash-grid encoding + MLP.
//
// The reference image is sampled on the fly on the GPU (bilinear filtering),
// a tiny-cuda-nn network regresses RGB colour from 2D coordinates, and stb is
// used to load the input image and to write the reference / learned images.

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <tiny-cuda-nn/trainer.h>

#include <cuda_runtime.h>

#include "cli.h"
#include "demo_config.h"
#include "image_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tcnn;

namespace {

using precision_t = network_precision_t;
using SteadyClock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr uint32_t kInputDims = 2;   // image coordinate (u, v)
constexpr uint32_t kOutputDims = 3;  // RGB

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 fetch_clamped(const float* image, int width, int height, int x, int y) {
    x = min(max(x, 0), width - 1);
    y = min(max(y, 0), height - 1);
    const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
    return make_float4(image[offset + 0], image[offset + 1], image[offset + 2], image[offset + 3]);
}

// Bilinear lookup into the reference image (RGBA float, clamped at the border).
__device__ __forceinline__ float4 sample_reference(const float* image, int width, int height, float u, float v) {
    const float x = u * width - 0.5f;
    const float y = v * height - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float fx = x - x0;
    const float fy = y - y0;

    const float4 a = fetch_clamped(image, width, height, x0, y0);
    const float4 b = fetch_clamped(image, width, height, x0 + 1, y0);
    const float4 c = fetch_clamped(image, width, height, x0, y0 + 1);
    const float4 d = fetch_clamped(image, width, height, x0 + 1, y0 + 1);

    float4 result;
    result.x = (a.x * (1.0f - fx) + b.x * fx) * (1.0f - fy) + (c.x * (1.0f - fx) + d.x * fx) * fy;
    result.y = (a.y * (1.0f - fx) + b.y * fx) * (1.0f - fy) + (c.y * (1.0f - fx) + d.y * fx) * fy;
    result.z = (a.z * (1.0f - fx) + b.z * fx) * (1.0f - fy) + (c.z * (1.0f - fx) + d.z * fx) * fy;
    result.w = 1.0f;
    return result;
}

__global__ void sample_reference_kernel(uint32_t n_samples,
                                        const float* __restrict__ coords,
                                        const float* __restrict__ image,
                                        int width, int height,
                                        float* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) {
        return;
    }
    const float4 rgb = sample_reference(image, width, height, coords[2 * i + 0], coords[2 * i + 1]);
    out[3 * i + 0] = rgb.x;
    out[3 * i + 1] = rgb.y;
    out[3 * i + 2] = rgb.z;
}

// Column-major network output (stride floats per pixel) -> interleaved RGB.
__global__ void matrix_to_rgb_kernel(uint32_t n_pixels,
                                     const float* __restrict__ matrix,
                                     uint32_t stride,
                                     float* __restrict__ rgb) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) {
        return;
    }
    rgb[3 * i + 0] = matrix[static_cast<size_t>(i) * stride + 0];
    rgb[3 * i + 1] = matrix[static_cast<size_t>(i) * stride + 1];
    rgb[3 * i + 2] = matrix[static_cast<size_t>(i) * stride + 2];
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

uint32_t round_up(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

void ensure_parent(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

double compute_psnr(const std::vector<float>& predicted_rgb, const demo::Image& reference) {
    double mse = 0.0;
    const size_t n_pixels = static_cast<size_t>(reference.width) * reference.height;
    for (size_t i = 0; i < n_pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float predicted = std::min(std::max(predicted_rgb[3 * i + c], 0.0f), 1.0f);
            const double diff = static_cast<double>(predicted) - reference.rgba[4 * i + c];
            mse += diff * diff;
        }
    }
    mse /= static_cast<double>(n_pixels * 3);
    if (mse <= 0.0) {
        return 99.0;
    }
    return 10.0 * std::log10(1.0 / mse);
}

std::vector<float> evaluate_network(NetworkWithInputEncoding<precision_t>& network,
                                    cudaStream_t stream,
                                    GPUMatrix<float>& input,
                                    GPUMatrix<float>& output,
                                    uint32_t n_pixels) {
    network.inference(stream, input, output);

    GPUMemory<float> rgb(static_cast<size_t>(n_pixels) * 3);
    linear_kernel(matrix_to_rgb_kernel, 0, stream, n_pixels, output.data(), output.stride(), rgb.data());

    std::vector<float> host(rgb.size());
    CUDA_CHECK_THROW(cudaMemcpyAsync(host.data(), rgb.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return host;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        demo::Options options = demo::parse_args(argc, argv);
        if (options.help) {
            demo::print_usage(argv[0]);
            return 0;
        }

        // --- Device ------------------------------------------------------
        int device_count = 0;
        CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA device found");
        }
        if (options.gpu < 0 || options.gpu >= device_count) {
            throw std::runtime_error("invalid --gpu " + std::to_string(options.gpu) + " (" +
                                     std::to_string(device_count) + " device(s) available)");
        }
        CUDA_CHECK_THROW(cudaSetDevice(options.gpu));

        cudaDeviceProp properties{};
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties, options.gpu));

        std::printf("=== tiny-cuda-nn image fitting demo ===\n");
        std::printf("GPU %d: %s (sm_%d%d, %.1f GiB, CUDA %d.%d)\n",
                    options.gpu, properties.name, properties.major, properties.minor,
                    properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                    CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

        const uint32_t compute_capability = cuda_compute_capability(options.gpu);
        if (compute_capability < MIN_GPU_ARCH) {
            std::printf("Warning: binary compiled for sm_%u+, device reports sm_%u.\n",
                        MIN_GPU_ARCH, compute_capability);
        }

        // --- Configuration ----------------------------------------------
        const std::string config_path = options.config_path.empty()
            ? std::string(TCNN_DEMO_SOURCE_DIR) + "/configs/image_hashgrid.json"
            : options.config_path;

        std::ifstream config_stream(config_path);
        if (!config_stream) {
            throw std::runtime_error("cannot open config file '" + config_path + "'");
        }
        json config = json::parse(config_stream, nullptr, true, /*ignore_comments=*/true);

        const json train_config = config.value("train", json::object());
        const uint32_t n_steps = options.steps.value_or(train_config.value("n_steps", 2000u));
        const uint32_t batch_size = round_up(
            options.batch_size.value_or(train_config.value("batch_size", 65536u)), BATCH_SIZE_GRANULARITY);
        const uint32_t save_every = options.save_every.value_or(train_config.value("save_every", 250u));
        const uint32_t seed = options.seed.value_or(train_config.value("seed", 1337u));
        if (n_steps == 0 || batch_size == 0) {
            throw std::runtime_error("--steps and --batch-size must be positive");
        }

        std::printf("Config: %s\n", config_path.c_str());

        // --- Reference image --------------------------------------------
        demo::Image reference;
        if (options.input_image.empty()) {
            reference = demo::make_test_pattern(options.width, options.height);
            std::printf("Reference: %dx%d synthetic test pattern (no --input given)\n",
                        reference.width, reference.height);
        } else {
            reference = demo::load_image(options.input_image);
            std::printf("Reference: %s (%dx%d)\n", options.input_image.c_str(), reference.width, reference.height);
        }
        if (reference.width < 2 || reference.height < 2) {
            throw std::runtime_error("reference image is too small");
        }

        const fs::path output_path(options.output_image);
        ensure_parent(output_path);
        fs::path output_dir = output_path.parent_path();
        if (output_dir.empty()) {
            output_dir = ".";
        }
        ensure_parent(output_dir / "reference.png");
        demo::save_image((output_dir / "reference.png").string(), reference.data(),
                         reference.width, reference.height, 3, 4, options.gamma);

        const fs::path preview_dir = output_dir / "previews";
        if (save_every > 0) {
            ensure_parent(preview_dir / "preview.png");
        }

        const uint32_t n_pixels = static_cast<uint32_t>(reference.width) * reference.height;
        const uint32_t n_pixels_padded = round_up(n_pixels, BATCH_SIZE_GRANULARITY);

        GPUMemory<float> gpu_reference(reference.rgba.size());
        gpu_reference.copy_from_host(reference.data(), reference.rgba.size());

        GPUMemory<float> coordinates(static_cast<size_t>(n_pixels_padded) * kInputDims);
        std::vector<float> host_coordinates(static_cast<size_t>(n_pixels_padded) * kInputDims, 0.0f);
        for (int y = 0; y < reference.height; ++y) {
            for (int x = 0; x < reference.width; ++x) {
                const size_t idx = (static_cast<size_t>(y) * reference.width + x) * kInputDims;
                host_coordinates[idx + 0] = (x + 0.5f) / reference.width;
                host_coordinates[idx + 1] = (y + 0.5f) / reference.height;
            }
        }
        coordinates.copy_from_host(host_coordinates.data(), host_coordinates.size());

        GPUMatrix<float> inference_input(coordinates.data(), kInputDims, n_pixels_padded);
        GPUMatrix<float> inference_output(kOutputDims, n_pixels_padded);

        cudaStream_t stream;
        CUDA_CHECK_THROW(cudaStreamCreate(&stream));

        // --- Model -------------------------------------------------------
        std::shared_ptr<Loss<precision_t>> loss(
            create_loss<precision_t>(config.value("loss", json::object())));
        std::shared_ptr<Optimizer<precision_t>> optimizer(
            create_optimizer<precision_t>(config.value("optimizer", json::object())));
        auto network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
            kInputDims, kOutputDims,
            config.value("encoding", json::object()),
            config.value("network", json::object()));

        // JIT fusion compiles kernels at runtime through NVRTC. Loading the resulting
        // PTX requires a driver at least as new as the CUDA toolkit; otherwise the
        // driver rejects it with CUDA_ERROR_UNSUPPORTED_PTX_VERSION, so disable JIT up
        // front instead of letting the first training step fail and log a warning.
        int driver_version = 0;
        int runtime_version = 0;
        CUDA_CHECK_THROW(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK_THROW(cudaRuntimeGetVersion(&runtime_version));

        const bool jit_supported = supports_jit_fusion(options.gpu);
        const bool driver_supports_ptx = driver_version >= runtime_version;
        const bool use_jit = !options.no_jit && jit_supported && driver_supports_ptx;
        network->set_jit_fusion(use_jit);

        std::printf("Model: %uD -> %uD, %zu trainable parameters\n",
                    kInputDims, kOutputDims, network->n_params());
        if (use_jit) {
            std::printf("JIT fusion: on\n");
        } else if (options.no_jit) {
            std::printf("JIT fusion: off (--no-jit)\n");
        } else if (!jit_supported) {
            std::printf("JIT fusion: off (GPU compute capability not supported)\n");
        } else {
            std::printf("JIT fusion: off (driver CUDA %d.%d < runtime %d.%d; update the driver to enable)\n",
                        driver_version / 1000, (driver_version % 1000) / 10,
                        runtime_version / 1000, (runtime_version % 1000) / 10);
        }

        auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
            network, optimizer, loss, seed);

        GPUMatrix<float> training_input(kInputDims, batch_size);
        GPUMatrix<float> training_target(kOutputDims, batch_size);
        default_rng_t rng{seed};

        auto training_step = [&]() {
            generate_random_uniform<float>(stream, rng, static_cast<size_t>(batch_size) * kInputDims,
                                           training_input.data());
            linear_kernel(sample_reference_kernel, 0, stream, batch_size,
                          training_input.data(), gpu_reference.data(), reference.width, reference.height,
                          training_target.data());
            return trainer->training_step(stream, training_input, training_target);
        };

        // Warm-up: first steps trigger JIT compilation and workspace allocation.
        for (int i = 0; i < 3; ++i) {
            training_step();
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        // --- Train -------------------------------------------------------
        std::printf("Training %u steps, batch size %u, %u pixels\n",
                    n_steps, batch_size, n_pixels);
        std::fflush(stdout);

        const auto begin = SteadyClock::now();
        const uint32_t report_every = std::max(1u, n_steps / 10);

        for (uint32_t step = 0; step < n_steps; ++step) {
            auto context = training_step();

            if (step % report_every == 0 || step + 1 == n_steps) {
                const float current_loss = trainer->loss(stream, *context);
                const auto now = SteadyClock::now();
                const double elapsed = std::chrono::duration<double>(now - begin).count();
                std::printf("[%7u/%u] loss %.6f  %.0f steps/s  (%.2fs)\n",
                            step + 1, n_steps, current_loss,
                            (step + 1) / std::max(elapsed, 1e-9), elapsed);
                std::fflush(stdout);
            }

            if (save_every > 0 && (step + 1) % save_every == 0) {
                const std::vector<float> rgb = evaluate_network(
                    *network, stream, inference_input, inference_output, n_pixels);
                char name[64];
                std::snprintf(name, sizeof(name), "step_%06u.png", step + 1);
                demo::save_image((preview_dir / name).string(), rgb.data(),
                                 reference.width, reference.height, 3, 3, options.gamma);
                std::printf("  wrote preview %s\n", (preview_dir / name).string().c_str());
                std::fflush(stdout);
            }
        }

        const auto end = SteadyClock::now();
        const double total_seconds = std::chrono::duration<double>(end - begin).count();

        // --- Final render -------------------------------------------------
        const std::vector<float> learned = evaluate_network(
            *network, stream, inference_input, inference_output, n_pixels);

        demo::save_image(output_path.string(), learned.data(),
                         reference.width, reference.height, 3, 3, options.gamma);
        demo::save_side_by_side((output_dir / "comparison.png").string(),
                                reference.data(), 4, learned.data(), 3,
                                reference.width, reference.height, 3, options.gamma);

        std::printf("Final PSNR: %.2f dB\n", compute_psnr(learned, reference));
        std::printf("Total time: %.2fs (%.0f steps/s)\n", total_seconds, n_steps / std::max(total_seconds, 1e-9));
        std::printf("Wrote: %s\n", output_path.string().c_str());
        std::printf("       %s\n", (output_dir / "comparison.png").string().c_str());
        std::printf("       %s\n", (output_dir / "reference.png").string().c_str());

        free_all_gpu_memory_arenas();
        CUDA_CHECK_THROW(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
