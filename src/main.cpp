
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
    uint32_t n_output_dims = 1;
    uint32_t batch_size = 256;
};

class ImageDataset
{
public:
    ImageDataset(std::string path)
    {
        m_data = stbi_load(path.c_str(), &m_width, &m_height, &m_channels, 0);
        if (m_data == nullptr)
        {
            std::cout << "Load image " << path << " failed" << std::endl;
        }
        m_pixelnums = m_width * m_height;
    }

    void generate_training_data(tcnn::GPUMatrix<float>& gpu_input, tcnn::GPUMatrix<float>& gpu_output)
    {
        static int iterator_num = 0;
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        std::vector<float> cpu_input(Config::get()->n_input_dims * Config::get()->batch_size);
        std::vector<float> cpu_output(Config::get()->n_output_dims * Config::get()->batch_size);

        for (int ind = 0; ind < Config::get()->batch_size; ind ++)
        {
            double xx = dis(gen);
            double yy = dis(gen);
            
            cpu_input[2 * ind] = xx;
            cpu_input[2 * ind + 1] = yy;
            

            int w = xx * m_width;
            int h = yy * m_height;
            cpu_output[ind] = static_cast<float>( (m_data[h * m_width + w]) / 255.f );
        }

        CUDA_CHECK_THROW(cudaMemcpy(gpu_input.data(), cpu_input.data(), cpu_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(gpu_output.data(), cpu_output.data(), cpu_output.size() * sizeof(float), cudaMemcpyHostToDevice));

        iterator_num ++;
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
    Model()
    {
        m_model = tcnn::create_from_config(Config::get()->n_input_dims, Config::get()->n_output_dims, m_config);
    }

    void train()
    {
        for ()
    }

    void inference();

    nlohmann::json m_config = {
	{"loss", {
		{"otype", "L2"}
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

    tcnn::TrainableModel m_model;
};

int main()
{


    return 0;
}
