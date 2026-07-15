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

static bool write_file(const std::string& path, const std::vector<float>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    return (bool)ofs;
}

static void compare(const char* name, const std::vector<float>& out, const std::string& ref_path)
{
    std::vector<char> ref_bytes;
    if (!read_file(ref_path, ref_bytes))
    {
        std::printf("%s ref read failed\n", name);
        return;
    }
    const float* ref = (const float*)ref_bytes.data();
    const size_t ref_count = ref_bytes.size() / sizeof(float);
    const size_t n = std::min(ref_count, out.size());
    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.0f;
    int argmax_out = -1;
    int argmax_ref = -1;
    float best_out = -3.4e38f;
    float best_ref = -3.4e38f;
    for (size_t i = 0; i < n; i++)
    {
        const float e = std::fabs(out[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
        if (out[i] > best_out) { best_out = out[i]; argmax_out = (int)i; }
        if (ref[i] > best_ref) { best_ref = ref[i]; argmax_ref = (int)i; }
    }
    if (n > 0)
    {
        mae /= (double)n;
        rmse = std::sqrt(rmse / (double)n);
    }
    std::printf("%s count=%zu mae=%.9g rmse=%.9g maxe=%.9g argmax_out=%d argmax_ref=%d\n",
                name, n, mae, rmse, maxe, argmax_out, argmax_ref);
}

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        std::fprintf(stderr, "Usage: %s <param> <bin> <input_f32> <seq_len> <hidden_dim> <hidden_ref> <logits_ref> <out_hidden> <out_logits>\n", argv[0]);
        return 2;
    }
    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string input_path = argv[3];
    const int seq_len = std::atoi(argv[4]);
    const int hidden_dim = std::atoi(argv[5]);
    const std::string hidden_ref = argv[6];
    const std::string logits_ref = argv[7];
    const std::string out_hidden_path = argv[8];
    const std::string out_logits_path = argv[9];

    std::vector<char> input_bytes;
    if (!read_file(input_path, input_bytes)) return 1;
    if (input_bytes.size() != (size_t)seq_len * hidden_dim * sizeof(float))
    {
        std::fprintf(stderr, "bad input size\n");
        return 1;
    }
    ncnn::Mat in(hidden_dim, seq_len, (size_t)4u, 1);
    std::memcpy(in.data, input_bytes.data(), input_bytes.size());

    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out0, out1;
    if (ex.extract("out0", out0) != 0 || ex.extract("out1", out1) != 0)
    {
        std::fprintf(stderr, "extract failed\n");
        return 1;
    }
    std::vector<float> hidden(out0.total());
    std::vector<float> logits(out1.total());
    std::copy((const float*)out0.data, (const float*)out0.data + out0.total(), hidden.begin());
    std::copy((const float*)out1.data, (const float*)out1.data + out1.total(), logits.begin());
    write_file(out_hidden_path, hidden);
    write_file(out_logits_path, logits);
    std::printf("out0 dims w=%d h=%d c=%d total=%zu\n", out0.w, out0.h, out0.c, out0.total());
    std::printf("out1 dims w=%d h=%d c=%d total=%zu\n", out1.w, out1.h, out1.c, out1.total());
    compare("hidden", hidden, hidden_ref);
    compare("logits", logits, logits_ref);
    return 0;
}
