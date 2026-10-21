#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/config.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <random>

class Config
{
public:
    static Config* get()
    {
        static Config* config = nullptr;
        if (!config)
        {
            config = new Config();
        }
        return config;
    }

    uint32_t n_input_dims = 2;
    uint32_t n_output_dims = 3;
    uint32_t batch_size = 65536;
    uint32_t n_training_steps = 10000;
};

class ImageDataset
{
public:
    ImageDataset(std::string path)
    {
        // Force 3 channels so every image has the same layout as the network output.
        m_data = stbi_load(path.c_str(), &m_width, &m_height, &m_channels, 3);
        m_channels = 3;
        if (m_data == nullptr)
        {
            std::cout << "Load image " << path << " failed" << std::endl;
        }
        m_pixelnums = m_width * m_height;
    }

    void generate_training_data(tcnn::GPUMatrix<float>& gpu_input, tcnn::GPUMatrix<float>& gpu_output)
    {
        const uint32_t batch = Config::get()->batch_size;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        std::vector<float> cpu_input(2 * batch);
        std::vector<float> cpu_output(3 * batch);

        for (uint32_t i = 0; i < batch; i++)
        {
            float xx = dis(gen);
            float yy = dis(gen);

            int w = std::min((int)(xx * m_width),  m_width  - 1);
            int h = std::min((int)(yy * m_height), m_height - 1);

            // Pixel addressing must include the channel stride; the old code
            // read a random byte from the middle of the image instead.
            const unsigned char* px = m_data + (size_t)(h * m_width + w) * m_channels;

            cpu_input[2 * i + 0] = xx;
            cpu_input[2 * i + 1] = yy;
            for (int c = 0; c < 3; c++)
            {
                cpu_output[3 * i + c] = px[c] / 255.f;
            }
        }

        CUDA_CHECK_THROW(cudaMemcpy(gpu_input.data(), cpu_input.data(), cpu_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(gpu_output.data(), cpu_output.data(), cpu_output.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    int m_width;
    int m_height;
    int m_channels;
    int m_pixelnums;
    unsigned char* m_data;
};

class Model
{
public:
    Model(ImageDataset* img_ds)
    : m_img_ds(img_ds)
    {
        m_model = tcnn::create_from_config(Config::get()->n_input_dims, Config::get()->n_output_dims, m_config);
    }

    void train()
    {
        for (uint32_t iterator = 0; iterator < Config::get()->n_training_steps; iterator++)
        {
            tcnn::GPUMatrix<float> input(Config::get()->n_input_dims, Config::get()->batch_size);
            tcnn::GPUMatrix<float> ground_truth(Config::get()->n_output_dims, Config::get()->batch_size);

            m_img_ds->generate_training_data(input, ground_truth);

            auto ctx = m_model.trainer->training_step(input, ground_truth);
            float loss = m_model.trainer->loss(*ctx);

            if (iterator % 100 == 0)
            {
                std::cout << "iteration=" << iterator << " loss=" << loss << std::endl;
            }
        }
    }

    void inference(tcnn::GPUMatrix<float>& gpu_input, tcnn::GPUMatrix<float>& gpu_output)
    {
        m_model.network->inference(nullptr, gpu_input, gpu_output);
    }

    nlohmann::json m_config = {
        {"loss", {
            {"otype", "RelativeL2"}
        }},
        {"optimizer", {
            {"otype", "Adam"},
            {"learning_rate", 1e-3},
        }},
        {"network", {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 2},
        }},
    };

    ImageDataset* m_img_ds;
    tcnn::TrainableModel m_model;
};

int main()
{
    ImageDataset img_ds("/home/cl/nerual-render/cuda-nn-demo/asset/albert.jpg");
    Model model(&img_ds);

    // train
    model.train();

    // inference
    const uint32_t gran = tcnn::BATCH_SIZE_GRANULARITY;
    const uint32_t num = (img_ds.m_pixelnums + gran - 1) / gran * gran;
    std::vector<float> cpu_input(2 * (size_t)num, 0.f);
    std::vector<float> cpu_pred(3 * (size_t)num, 0.f);

    const float inv_w = 1.0f / (float)img_ds.m_width;
    const float inv_h = 1.0f / (float)img_ds.m_height;
    for (int h = 0; h < img_ds.m_height; h++)
    {
        for (int w = 0; w < img_ds.m_width; w++)
        {
            cpu_input[2 * (size_t)(h * img_ds.m_width + w)]     = (w + 0.5f) * inv_w;
            cpu_input[2 * (size_t)(h * img_ds.m_width + w) + 1] = (h + 0.5f) * inv_h;
        }
    }

    tcnn::GPUMatrix<float> gpu_input(2, num);
    tcnn::GPUMatrix<float> gpu_output(3, num);
    CUDA_CHECK_THROW(cudaMemcpy(gpu_input.data(), cpu_input.data(), cpu_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    model.inference(gpu_input, gpu_output);
    CUDA_CHECK_THROW(cudaMemcpy(cpu_pred.data(), gpu_output.data(), cpu_pred.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // float [0,1] -> 8-bit, clamp + round
    std::vector<unsigned char> final_res((size_t)num * 3);
    for (size_t ind = 0; ind < num; ind++)
    {
        for (int c = 0; c < 3; c++)
        {
            const float v = std::min(std::max(cpu_pred[ind * 3 + c], 0.0f), 1.0f);
            final_res[ind * 3 + c] = (unsigned char)(v * 255.0f + 0.5f);
        }
    }

    if (!stbi_write_png("/home/cl/nerual-render/cuda-nn-demo/asset/pred.png",
                        img_ds.m_width, img_ds.m_height, 3, final_res.data(), img_ds.m_width * 3)) {
        std::cerr << "failed to write png" << std::endl;
    }

    return 0;
}