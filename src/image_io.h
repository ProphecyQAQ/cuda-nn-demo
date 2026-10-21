// Minimal image I/O for the tiny-cuda-nn demo, built on top of stb_image and
// stb_image_write. Only 8-bit images are supported (PNG / JPG / BMP / TGA ...).
#pragma once

#include <string>
#include <vector>

namespace demo {

// Simple RGBA8 image representation converted to float [0, 1] on load.
// Row 0 is the top row, matching stb_image and PNG conventions.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgba;  // width * height * 4, interleaved RGBA

    float* data() { return rgba.data(); }
    const float* data() const { return rgba.data(); }
};

// Loads an image through stb_image and converts it to RGBA float.
// Throws std::runtime_error if the file cannot be read.
Image load_image(const std::string& path);

// Builds a deterministic, high-frequency test pattern (colour wheel + rings +
// checkerboard + thin grid). Used when no input image is given so that the
// demo can be run out of the box.
Image make_test_pattern(int width, int height);

// Writes an interleaved image to disk. The format is chosen from the file
// extension (.png, .jpg/.jpeg, .bmp, .tga).
//   pixels  : interleaved float pixels
//   channels: floats stored per pixel (1..4)
//   stride  : floats between the first channel of two neighbouring pixels
//   gamma   : display gamma applied as pow(v, 1 / gamma); 1.0 keeps values as-is
void save_image(const std::string& path, const float* pixels, int width, int height,
                int channels = 3, int stride = 3, float gamma = 1.0f);

// Writes two images of equal size side by side with a dark separator.
void save_side_by_side(const std::string& path,
                       const float* left, int left_stride,
                       const float* right, int right_stride,
                       int width, int height, int channels = 3,
                       float gamma = 1.0f, int separator = 4);

}  // namespace demo
