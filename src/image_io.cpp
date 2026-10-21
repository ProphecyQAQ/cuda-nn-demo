#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

// stb is third-party C code; keep our -Wall -Wextra build clean.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace demo {
namespace {

constexpr float kPi = 3.14159265358979323846f;

float apply_gamma(float value, float gamma) {
    value = std::min(std::max(value, 0.0f), 1.0f);
    return gamma == 1.0f ? value : std::pow(value, 1.0f / gamma);
}

std::string lower_extension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    std::string ext = path.substr(dot);
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes,
                 int width, int height, int channels) {
    const std::string ext = lower_extension(path);
    int ok = 0;
    if (ext == ".png") {
        ok = stbi_write_png(path.c_str(), width, height, channels, bytes.data(), width * channels);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        ok = stbi_write_jpg(path.c_str(), width, height, channels, bytes.data(), 95);
    } else if (ext == ".bmp") {
        ok = stbi_write_bmp(path.c_str(), width, height, channels, bytes.data());
    } else if (ext == ".tga") {
        ok = stbi_write_tga(path.c_str(), width, height, channels, bytes.data());
    } else {
        throw std::runtime_error("unsupported output image extension '" + ext +
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

}  // namespace

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

            const size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            image.rgba[idx + 0] = r;
            image.rgba[idx + 1] = g;
            image.rgba[idx + 2] = b;
            image.rgba[idx + 3] = 1.0f;
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
            const float* src = pixels + (static_cast<size_t>(y) * width + x) * stride;
            uint8_t* dst = bytes.data() + (static_cast<size_t>(y) * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[c] = static_cast<uint8_t>(std::lround(apply_gamma(src[c], gamma) * 255.0f));
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
            uint8_t* dst = bytes.data() + (static_cast<size_t>(y) * total_width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[c] = static_cast<uint8_t>(std::lround(apply_gamma(l[c], gamma) * 255.0f));
            }
            dst = bytes.data() + (static_cast<size_t>(y) * total_width + width + separator + x) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[c] = static_cast<uint8_t>(std::lround(apply_gamma(r[c], gamma) * 255.0f));
            }
        }
    }
    write_bytes(path, bytes, total_width, height, channels);
}

}  // namespace demo
