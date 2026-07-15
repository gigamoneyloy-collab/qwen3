#include "net.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static bool read_file(const std::string& path, std::vector<char>& data)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize((size_t)size);
    if (size > 0)
        ifs.read(data.data(), size);
    return (bool)ifs;
}

static bool write_file(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;
    ofs.write((const char*)data, (std::streamsize)bytes);
    return (bool)ofs;
}

int main(int argc, char** argv)
{
    if (argc != 8)
    {
        std::fprintf(stderr,
                     "Usage: %s <param> <bin> <codes_i32_bin> <code_len> <codebooks> <ref_f32_bin> <out_f32_bin>\n",
                     argv[0]);
        return 2;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string codes_path = argv[3];
    const int code_len = std::atoi(argv[4]);
    const int codebooks = std::atoi(argv[5]);
    const std::string ref_path = argv[6];
    const std::string out_path = argv[7];

    std::vector<char> code_bytes;
    if (!read_file(codes_path, code_bytes))
    {
        std::fprintf(stderr, "failed to read %s\n", codes_path.c_str());
        return 1;
    }
    const size_t expected_code_bytes = (size_t)code_len * codebooks * sizeof(int32_t);
    if (code_bytes.size() != expected_code_bytes)
    {
        std::fprintf(stderr, "unexpected code size %zu, expected %zu\n", code_bytes.size(), expected_code_bytes);
        return 1;
    }

    const int32_t* codes_tq = (const int32_t*)code_bytes.data(); // [T, Q]
    ncnn::Mat in(code_len, codebooks, 1, (size_t)4u, 1);        // [c=1, h=Q, w=T]
    for (int q = 0; q < codebooks; q++)
    {
        int32_t* row = in.channel(0).row<int32_t>(q);
        for (int t = 0; t < code_len; t++)
            row[t] = codes_tq[t * codebooks + q];
    }

    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;

    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load ncnn model\n");
        return 1;
    }

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", in) != 0)
    {
        std::fprintf(stderr, "failed to set input\n");
        return 1;
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
    {
        std::fprintf(stderr, "failed to extract out0\n");
        return 1;
    }

    const size_t out_count = out.total();
    std::vector<float> out_f32(out_count);
    const float* out_ptr = (const float*)out.data;
    std::copy(out_ptr, out_ptr + out_count, out_f32.begin());

    if (!write_file(out_path, out_f32.data(), out_f32.size() * sizeof(float)))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }

    std::vector<char> ref_bytes;
    if (!read_file(ref_path, ref_bytes))
    {
        std::fprintf(stderr, "failed to read %s\n", ref_path.c_str());
        return 1;
    }
    const size_t ref_count = ref_bytes.size() / sizeof(float);
    const float* ref = (const float*)ref_bytes.data();
    const size_t n = std::min(ref_count, out_count);

    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        const float e = std::fabs(out_f32[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
    }
    if (n > 0)
    {
        mae /= (double)n;
        rmse = std::sqrt(rmse / (double)n);
    }

    std::printf("out dims: w=%d h=%d c=%d total=%zu\n", out.w, out.h, out.c, out_count);
    std::printf("ref_count=%zu compare_count=%zu\n", ref_count, n);
    std::printf("mae=%.9g rmse=%.9g maxe=%.9g\n", mae, rmse, maxe);
    std::printf("wrote %s\n", out_path.c_str());

    return 0;
}
