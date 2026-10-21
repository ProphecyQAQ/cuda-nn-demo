// tiny-cuda-nn demo -- fit a 2D image with a hash-grid encoding + MLP.
//
// This is the only source file of the project. It contains
//   * image loading / saving through stb_image and stb_image_write
//   * a small command-line parser
//   * one tiny CUDA kernel that samples the reference image on the GPU
//   * tiny-cuda-nn training and inference
//
// CMake compiles this .cpp as CUDA source (see set_source_files_properties in
// CMakeLists.txt): tiny-cuda-nn only defines `network_precision_t` when
// __CUDACC__ is set, and the file contains a __global__ kernel.

#include <cuda_runtime.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <tiny-cuda-nn/trainer.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_WRITE_IMPLEMENTATION

// stb is third-party C code; keep our -Wall -Wextra build clean.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(__CUDACC__)
#pragma nv_diag_suppress 550
#endif
#include <stb_image.h>
#include <stb_image_write.h>
#if defined(__CUDACC__)
#pragma nv_diag_default 550
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tcnn;
namespace fs = std::filesystem;

namespace {

using precision_t = network_precision_t;
using SteadyClock = std::chrono::steady_clock;

constexpr uint32_t kInputDims = 2;   // image coordinate (u, v)
constexpr uint32_t kOutputDims = 3;  // RGB
constexpr float kPi = 3.14159265358979323846f;

// ===========================================================================
// GPU sampling
// ===========================================================================

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

// One thread per training sample: coordinates -> RGB target colour.
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

// ===========================================================================
// Image I/O (stb_image / stb_image_write)
// ===========================================================================

// RGBA8 image converted to float [0, 1]. Row 0 is the top row.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgba;  // width * height * 4

    float* data() { return rgba.data(); }
};

float apply_gamma(float value, float gamma) {
    value = std::min(std::max(value, 0.0f), 1.0f);
    return gamma == 1.0f ? value : std::pow(value, 1.0f / gamma);
}

std::string lower_extension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    std::string extension = path.substr(dot);
    for (char& c : extension) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return extension;
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes,
                 int width, int height, int channels) {
    const std::string extension = lower_extension(path);
    int ok = 0;
    if (extension == ".png") {
        ok = stbi_write_png(path.c_str(), width, height, channels, bytes.data(), width * channels);
    } else if (extension == ".jpg" || extension == ".jpeg") {
        ok = stbi_write_jpg(path.c_str(), width, height, channels, bytes.data(), 95);
    } else if (extension == ".bmp") {
        ok = stbi_write_bmp(path.c_str(), width, height, channels, bytes.data());
    } else if (extension == ".tga") {
        ok = stbi_write_tga(path.c_str(), width, height, channels, bytes.data());
    } else {
        throw std::runtime_error("unsupported output image extension '" + extension +
                                 "' (expected .png, .jpg, .bmp or .tga)");
    }
    if (!ok) {
        throw std::runtime_error("stb_image_write failed to write '" + path + "'");
    }
}

void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    h -= std::floor(h);
    const float sector = std::floor(h * 6.0f);
    const float f = h * 6.0f - sector;
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - f * s);
    const float t = v * (1.0f - (1.0f - f) * s);
    switch (static_cast<int>(sector) % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

Image load_image(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels_in_file, 4);
    if (!pixels) {
        throw std::runtime_error("failed to load image '" + path + "': " + stbi_failure_reason());
    }

    Image image;
    image.width = width;
    image.height = height;
    image.rgba.resize(static_cast<size_t>(width) * height * 4);
    for (size_t i = 0; i < image.rgba.size(); ++i) {
        image.rgba[i] = pixels[i] / 255.0f;
    }
    stbi_image_free(pixels);
    return image;
}

Image make_test_pattern(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("test pattern size must be positive");
    }

    Image image;
    image.width = width;
    image.height = height;
    image.rgba.resize(static_cast<size_t>(width) * height * 4);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float u = (x + 0.5f) / width;
            const float v = (y + 0.5f) / height;
            const float dx = u - 0.5f;
            const float dy = v - 0.5f;
            const float radius = std::sqrt(dx * dx + dy * dy);

            // Hue wheel: maximum colour variety across the image.
            const float hue = std::atan2(dy, dx) / (2.0f * kPi) + 0.5f;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            hsv_to_rgb(hue, 0.85f, 1.0f, r, g, b);

            // Radial rings and a checkerboard: smooth and sharp features.
            const float rings = 0.55f + 0.45f * (0.5f + 0.5f * std::cos(2.0f * kPi * 10.0f * radius));
            const float checker = ((x / 32 + y / 32) & 1) ? 1.0f : 0.62f;
            r *= rings * checker;
            g *= rings * checker;
            b *= rings * checker;

            // Thin bright grid lines: high-frequency detail for the hash grid.
            if (x % 64 == 0 || y % 64 == 0) {
                r = 0.85f + 0.15f * r;
                g = 0.85f + 0.15f * g;
                b = 0.85f + 0.15f * b;
            }

            const size_t index = (static_cast<size_t>(y) * width + x) * 4;
            image.rgba[index + 0] = r;
            image.rgba[index + 1] = g;
            image.rgba[index + 2] = b;
            image.rgba[index + 3] = 1.0f;
        }
    }
    return image;
}

void save_image(const std::string& path, const float* pixels, int width, int height,
                int channels, int stride, float gamma) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("cannot save an image with non-positive size");
    }
    if (channels < 1 || channels > 4) {
        throw std::runtime_error("save_image supports 1..4 channels");
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(width) * height * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float* source = pixels + (static_cast<size_t>(y) * width + x) * stride;
            uint8_t* destination = bytes.data() + (static_cast<size_t>(y) * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                destination[c] = static_cast<uint8_t>(std::lround(apply_gamma(source[c], gamma) * 255.0f));
            }
        }
    }
    write_bytes(path, bytes, width, height, channels);
}

void save_side_by_side(const std::string& path,
                       const float* left, int left_stride,
                       const float* right, int right_stride,
                       int width, int height, int channels,
                       float gamma, int separator) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("cannot save an image with non-positive size");
    }
    if (separator < 0) {
        separator = 0;
    }

    const int total_width = width * 2 + separator;
    std::vector<uint8_t> bytes(static_cast<size_t>(total_width) * height * channels, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float* l = left + (static_cast<size_t>(y) * width + x) * left_stride;
            const float* r = right + (static_cast<size_t>(y) * width + x) * right_stride;
            uint8_t* destination = bytes.data() + (static_cast<size_t>(y) * total_width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                destination[c] = static_cast<uint8_t>(std::lround(apply_gamma(l[c], gamma) * 255.0f));
            }
            destination = bytes.data() + (static_cast<size_t>(y) * total_width + width + separator + x) * channels;
            for (int c = 0; c < channels; ++c) {
                destination[c] = static_cast<uint8_t>(std::lround(apply_gamma(r[c], gamma) * 255.0f));
            }
        }
    }
    write_bytes(path, bytes, total_width, height, channels);
}

// ===========================================================================
// Command line
// ===========================================================================

struct Options {
    std::string input_image;                      // empty -> procedural test pattern
    std::string output_image = "output/learned.png";
    std::string config_path;                      // empty -> configs/image_hashgrid.json
    std::optional<uint32_t> steps;
    std::optional<uint32_t> batch_size;
    std::optional<uint32_t> save_every;
    std::optional<uint32_t> seed;
    int gpu = 0;
    int width = 512;
    int height = 512;
    float gamma = 1.0f;
    bool no_jit = false;
    bool help = false;
};

uint32_t parse_u32(const std::string& option, const std::string& value) {
    try {
        size_t position = 0;
        const unsigned long parsed = std::stoul(value, &position);
        if (position != value.size() || parsed > 0xffffffffUL) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<uint32_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

int parse_int(const std::string& option, const std::string& value) {
    try {
        size_t position = 0;
        const long parsed = std::stol(value, &position);
        if (position != value.size() || parsed < -2147483648L || parsed > 2147483647L) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

float parse_float(const std::string& option, const std::string& value) {
    try {
        size_t position = 0;
        const float parsed = std::stof(value, &position);
        if (position != value.size()) {
            throw std::invalid_argument("bad value");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string inline_value;
        bool has_inline_value = false;

        if (arg.rfind("--", 0) == 0) {
            const auto equals = arg.find('=');
            if (equals != std::string::npos) {
                inline_value = arg.substr(equals + 1);
                arg = arg.substr(0, equals);
                has_inline_value = true;
            }
        }

        auto value = [&](const std::string& option) -> std::string {
            if (has_inline_value) {
                return inline_value;
            }
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for option '" + option + "'");
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            options.help = true;
        } else if (arg == "-i" || arg == "--input") {
            options.input_image = value(arg);
        } else if (arg == "-o" || arg == "--output") {
            options.output_image = value(arg);
        } else if (arg == "-c" || arg == "--config") {
            options.config_path = value(arg);
        } else if (arg == "-n" || arg == "--steps") {
            options.steps = parse_u32(arg, value(arg));
        } else if (arg == "-b" || arg == "--batch-size") {
            options.batch_size = parse_u32(arg, value(arg));
        } else if (arg == "--save-every") {
            options.save_every = parse_u32(arg, value(arg));
        } else if (arg == "--seed") {
            options.seed = parse_u32(arg, value(arg));
        } else if (arg == "--gpu") {
            options.gpu = parse_int(arg, value(arg));
        } else if (arg == "--width") {
            options.width = parse_int(arg, value(arg));
        } else if (arg == "--height") {
            options.height = parse_int(arg, value(arg));
        } else if (arg == "--gamma") {
            options.gamma = parse_float(arg, value(arg));
        } else if (arg == "--no-jit") {
            options.no_jit = true;
        } else {
            throw std::runtime_error("unknown option '" + arg + "' (see --help)");
        }
    }

    return options;
}

void print_usage(const char* argv0) {
    std::printf(
        "tiny-cuda-nn demo: fit a 2D image with a hash-grid encoded MLP.\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "  -i, --input PATH      image to learn (PNG/JPG/BMP/TGA).\n"
        "                        Without it, a synthetic test pattern is used.\n"
        "  -o, --output PATH     output image, .png by default [output/learned.png]\n"
        "  -c, --config PATH     JSON config [configs/image_hashgrid.json]\n"
        "  -n, --steps N         number of training steps [from config: 2000]\n"
        "  -b, --batch-size N    samples per step, rounded up to 256 [65536]\n"
        "      --save-every N    write a preview every N steps (0 = off) [250]\n"
        "      --seed N          RNG seed [1337]\n"
        "      --gpu N           CUDA device index [0]\n"
        "      --width N         test-pattern width [512]\n"
        "      --height N        test-pattern height [512]\n"
        "      --gamma G         display gamma used when writing images [1.0]\n"
        "      --no-jit          disable tiny-cuda-nn JIT fusion\n"
        "  -h, --help            show this help\n",
        argv0);
}

// ===========================================================================
// Misc helpers
// ===========================================================================

uint32_t round_up(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

void ensure_parent(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

std::string default_config_path(const char* argv0) {
    std::error_code error;
    const fs::path executable = fs::absolute(fs::path(argv0), error);
    if (!error) {
        // The executable normally lives in <project>/build/tcnn_demo.
        const fs::path candidate = executable.parent_path().parent_path() / "configs" / "image_hashgrid.json";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    return "configs/image_hashgrid.json";
}

double compute_psnr(const std::vector<float>& predicted_rgb, const Image& reference) {
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

// Runs inference on n_pixels samples and returns interleaved RGB on the host.
// A column-major [3 x n] GPUMatrix is already laid out as [pixel][channel].
std::vector<float> evaluate_network(NetworkWithInputEncoding<precision_t>& network,
                                    cudaStream_t stream,
                                    GPUMatrix<float>& input,
                                    GPUMatrix<float>& output,
                                    uint32_t n_pixels) {
    network.inference(stream, input, output);

    std::vector<float> host(static_cast<size_t>(output.stride()) * output.n());
    CUDA_CHECK_THROW(cudaMemcpyAsync(host.data(), output.data(), host.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    host.resize(static_cast<size_t>(n_pixels) * 3);
    return host;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        if (options.help) {
            print_usage(argv[0]);
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
            ? default_config_path(argv[0])
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
        Image reference;
        if (options.input_image.empty()) {
            reference = make_test_pattern(options.width, options.height);
            std::printf("Reference: %dx%d synthetic test pattern (no --input given)\n",
                        reference.width, reference.height);
        } else {
            reference = load_image(options.input_image);
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
        save_image((output_dir / "reference.png").string(), reference.data(),
                   reference.width, reference.height, 3, 4, options.gamma);

        const fs::path preview_dir = output_dir / "previews";
        if (save_every > 0) {
            ensure_parent(preview_dir / "preview.png");
        }

        const uint32_t n_pixels = static_cast<uint32_t>(reference.width) * reference.height;
        const uint32_t n_pixels_padded = round_up(n_pixels, BATCH_SIZE_GRANULARITY);

        GPUMemory<float> gpu_reference(reference.rgba.size());
        gpu_reference.copy_from_host(reference.data(), reference.rgba.size());

        // Full-resolution coordinate grid used for previews and the final image.
        GPUMemory<float> inference_coordinates(static_cast<size_t>(n_pixels_padded) * kInputDims);
        std::vector<float> host_inference_coordinates(static_cast<size_t>(n_pixels_padded) * kInputDims, 0.0f);
        for (int y = 0; y < reference.height; ++y) {
            for (int x = 0; x < reference.width; ++x) {
                const size_t index = (static_cast<size_t>(y) * reference.width + x) * kInputDims;
                host_inference_coordinates[index + 0] = (x + 0.5f) / reference.width;
                host_inference_coordinates[index + 1] = (y + 0.5f) / reference.height;
            }
        }
        inference_coordinates.copy_from_host(host_inference_coordinates.data(), host_inference_coordinates.size());

        GPUMatrix<float> inference_input(inference_coordinates.data(), kInputDims, n_pixels_padded);
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
            // Fresh random coordinates every step...
            generate_random_uniform<float>(stream, rng, static_cast<size_t>(batch_size) * kInputDims,
                                           training_input.data());
            // ...and their colours, sampled from the reference image on the GPU.
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
                save_image((preview_dir / name).string(), rgb.data(),
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

        save_image(output_path.string(), learned.data(),
                   reference.width, reference.height, 3, 3, options.gamma);
        save_side_by_side((output_dir / "comparison.png").string(),
                          reference.data(), 4, learned.data(), 3,
                          reference.width, reference.height, 3, options.gamma, 4);

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
