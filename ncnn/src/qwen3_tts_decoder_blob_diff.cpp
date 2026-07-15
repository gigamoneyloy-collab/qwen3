#include "net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_file(const std::string& path, std::vector<char>& data)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize((size_t)size);
    if (size > 0) ifs.read(data.data(), size);
    return (bool)ifs;
}

static bool read_f32(const std::string& path, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(float) != 0) return false;
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static ncnn::Mat make_decoder_input(const std::vector<float>& hidden, int hidden_frames, int chunk_frames, int dim)
{
    std::vector<float> chunk((size_t)chunk_frames * dim, 0.0f);
    const int copy_frames = std::min(hidden_frames, chunk_frames);
    std::memcpy(chunk.data(), hidden.data(), (size_t)copy_frames * dim * sizeof(float));
    ncnn::Mat in(dim, chunk_frames, (size_t)4u, 1);
    std::memcpy(in.data, chunk.data(), chunk.size() * sizeof(float));
    return in;
}

static bool extract_blob(const std::string& param,
                         const std::string& bin,
                         const ncnn::Mat& in,
                         const std::string& blob,
                         bool use_vulkan,
                         int threads,
                         ncnn::Mat& out)
{
    ncnn::Net net;
    net.opt.num_threads = std::max(1, threads);
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load decoder\n");
        return false;
    }

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", in) != 0)
    {
        std::fprintf(stderr, "failed to set input\n");
        return false;
    }
    if (ex.extract(blob.c_str(), out) != 0)
    {
        std::fprintf(stderr, "failed to extract blob %s\n", blob.c_str());
        return false;
    }
    return true;
}

static void compare_mat(const ncnn::Mat& cpu, const ncnn::Mat& gpu, int probe_idx)
{
    const size_t n = std::min(cpu.total(), gpu.total());
    const float* a = (const float*)cpu.data;
    const float* b = (const float*)gpu.data;
    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.0f;
    int maxidx = -1;
    for (size_t i = 0; i < n; i++)
    {
        const float e = std::fabs(a[i] - b[i]);
        mae += e;
        rmse += (double)e * e;
        if (e > maxe)
        {
            maxe = e;
            maxidx = (int)i;
        }
    }
    if (n)
    {
        mae /= (double)n;
        rmse = std::sqrt(rmse / (double)n);
    }
    std::printf("cpu shape dims=%d w=%d h=%d c=%d total=%zu elemsize=%zu elempack=%d\n",
                cpu.dims, cpu.w, cpu.h, cpu.c, cpu.total(), cpu.elemsize, cpu.elempack);
    std::printf("gpu shape dims=%d w=%d h=%d c=%d total=%zu elemsize=%zu elempack=%d\n",
                gpu.dims, gpu.w, gpu.h, gpu.c, gpu.total(), gpu.elemsize, gpu.elempack);
    std::printf("compare_count=%zu cpu_total=%zu gpu_total=%zu\n", n, cpu.total(), gpu.total());
    std::printf("mae=%.9g rmse=%.9g maxe=%.9g maxidx=%d\n", mae, rmse, maxe, maxidx);
    if (maxidx >= 0)
        std::printf("at_max cpu=%.9g gpu=%.9g\n", a[maxidx], b[maxidx]);
    if (probe_idx >= 0 && (size_t)probe_idx < n)
    {
        std::printf("probe idx=%d cpu=%.9g gpu=%.9g diff=%.9g\n",
                    probe_idx, a[probe_idx], b[probe_idx], std::fabs(a[probe_idx] - b[probe_idx]));
    }
}

int main(int argc, char** argv)
{
    if (argc != 9 && argc != 10)
    {
        std::fprintf(stderr, "Usage: %s <decoder_param> <decoder_bin> <hidden_f32> <hidden_frames> <chunk_frames> <hidden_dim> <blob_name> <threads> [probe_idx]\n", argv[0]);
        return 2;
    }

    const std::string param = argv[1];
    const std::string bin = argv[2];
    const std::string hidden_path = argv[3];
    const int hidden_frames = std::atoi(argv[4]);
    const int chunk_frames = std::atoi(argv[5]);
    const int dim = std::atoi(argv[6]);
    const std::string blob = argv[7];
    const int threads = std::atoi(argv[8]);
    const int probe_idx = argc == 10 ? std::atoi(argv[9]) : -1;

    std::vector<float> hidden;
    if (!read_f32(hidden_path, hidden) || hidden.size() != (size_t)hidden_frames * dim)
    {
        std::fprintf(stderr, "bad hidden file %s size=%zu expected=%zu\n",
                     hidden_path.c_str(), hidden.size(), (size_t)hidden_frames * dim);
        return 1;
    }

    ncnn::Mat in = make_decoder_input(hidden, hidden_frames, chunk_frames, dim);
    ncnn::Mat cpu;
    ncnn::Mat gpu;
    if (!extract_blob(param, bin, in, blob, false, threads, cpu)) return 1;
    if (!extract_blob(param, bin, in, blob, true, threads, gpu)) return 1;
    compare_mat(cpu, gpu, probe_idx);
    return 0;
}
