# stb (vendored)

`stb_image.h` and `stb_image_write.h` are vendored unmodified from
<https://github.com/nothings/stb> at commit
`a7d4c2175630e1ea5b6f8dd00ab96fed0b97b96d`.

They are used for reading the input image and writing the reference /
learned / comparison images. Only 8-bit formats are supported (PNG, JPG,
BMP, TGA, ...); HDR/EXR is not.

License: dual-licensed public domain (Unlicense) / MIT -- see the end of
each header file.
