// Tiny command-line parser for the demo. Header-only, no external deps.
#pragma once

#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

namespace demo {

struct Options {
    std::string input_image;                      // empty -> procedural test pattern
    std::string output_image = "output/learned.png";
    std::string config_path;                      // empty -> configs/image_hashgrid.json
    std::optional<uint32_t> steps;                // 0 -> value from the config file
    std::optional<uint32_t> batch_size;           // 0 -> value from the config file
    std::optional<uint32_t> save_every;           // 0 -> disabled
    std::optional<uint32_t> seed;
    int gpu = 0;
    int width = 512;                              // test pattern size
    int height = 512;
    float gamma = 1.0f;
    bool no_jit = false;
    bool help = false;
};

namespace detail {

inline uint32_t parse_u32(const std::string& option, const std::string& value) {
    try {
        size_t pos = 0;
        const unsigned long parsed = std::stoul(value, &pos);
        if (pos != value.size() || parsed > 0xffffffffUL) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<uint32_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

inline int parse_int(const std::string& option, const std::string& value) {
    try {
        size_t pos = 0;
        const long parsed = std::stol(value, &pos);
        if (pos != value.size() || parsed < -2147483648L || parsed > 2147483647L) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

inline float parse_float(const std::string& option, const std::string& value) {
    try {
        size_t pos = 0;
        const float parsed = std::stof(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("bad value");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + option + ": '" + value + "'");
    }
}

}  // namespace detail

inline Options parse_args(int argc, char** argv) {
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
            options.steps = detail::parse_u32(arg, value(arg));
        } else if (arg == "-b" || arg == "--batch-size") {
            options.batch_size = detail::parse_u32(arg, value(arg));
        } else if (arg == "--save-every") {
            options.save_every = detail::parse_u32(arg, value(arg));
        } else if (arg == "--seed") {
            options.seed = detail::parse_u32(arg, value(arg));
        } else if (arg == "--gpu") {
            options.gpu = detail::parse_int(arg, value(arg));
        } else if (arg == "--width") {
            options.width = detail::parse_int(arg, value(arg));
        } else if (arg == "--height") {
            options.height = detail::parse_int(arg, value(arg));
        } else if (arg == "--gamma") {
            options.gamma = detail::parse_float(arg, value(arg));
        } else if (arg == "--no-jit") {
            options.no_jit = true;
        } else {
            throw std::runtime_error("unknown option '" + arg + "' (see --help)");
        }
    }

    return options;
}

inline void print_usage(const char* argv0) {
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

}  // namespace demo
