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

static ncnn::Mat mat_from_f32(const std::string& path, int w, int h, int c)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != (size_t)w * h * c * sizeof(float))
    {
        std::fprintf(stderr, "bad input file %s\n", path.c_str());
        std::exit(1);
    }
    ncnn::Mat m;
    if (c > 1) m.create(w, h, c, (size_t)4u, 1);
    else m.create(w, h, (size_t)4u, 1);
    std::memcpy(m.data, bytes.data(), bytes.size());
    return m;
}

static ncnn::Mat mat_from_f32_4d(const std::string& path, int w, int h, int d, int c)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != (size_t)w * h * d * c * sizeof(float))
    {
        std::fprintf(stderr, "bad input file %s\n", path.c_str());
        std::exit(1);
    }
    ncnn::Mat m;
    m.create(w, h, d, c, (size_t)4u, 1);
    std::memcpy(m.data, bytes.data(), bytes.size());
    return m;
}

static ncnn::Mat rope_mat_from_f32(const std::string& path, int seq_len)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes))
    {
        std::fprintf(stderr, "bad input file %s\n", path.c_str());
        std::exit(1);
    }
    if (bytes.size() == (size_t)128 * seq_len * sizeof(float))
    {
        ncnn::Mat m;
        m.create(128, seq_len, 1, 1, (size_t)4u, 1);
        std::memcpy(m.data, bytes.data(), bytes.size());
        return m;
    }
    if (bytes.size() == (size_t)128 * seq_len * 3 * sizeof(float))
    {
        ncnn::Mat m;
        m.create(128, seq_len, 1, 3, (size_t)4u, 1);
        std::memcpy(m.data, bytes.data(), bytes.size());
        return m;
    }
    std::fprintf(stderr, "bad rope file %s size=%zu\n", path.c_str(), bytes.size());
    std::exit(1);
}

static void compare(const char* name, const ncnn::Mat& out, const std::string& ref_path)
{
    std::vector<char> ref_bytes;
    if (!read_file(ref_path, ref_bytes))
    {
        std::printf("%s ref read failed\n", name);
        return;
    }
    const float* ref = (const float*)ref_bytes.data();
    const float* p = (const float*)out.data;
    const size_t n = std::min(ref_bytes.size() / sizeof(float), out.total());
    double mae = 0.0, rmse = 0.0;
    float maxe = 0.0f;
    int argmax_out = -1, argmax_ref = -1;
    float best_out = -3.4e38f, best_ref = -3.4e38f;
    for (size_t i = 0; i < n; i++)
    {
        const float e = std::fabs(p[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
        if (p[i] > best_out) { best_out = p[i]; argmax_out = (int)i; }
        if (ref[i] > best_ref) { best_ref = ref[i]; argmax_ref = (int)i; }
    }
    if (n) { mae /= (double)n; rmse = std::sqrt(rmse / (double)n); }
    std::printf("%s count=%zu mae=%.9g rmse=%.9g maxe=%.9g argmax_out=%d argmax_ref=%d\n",
                name, n, mae, rmse, maxe, argmax_out, argmax_ref);
}

int main(int argc, char** argv)
{
    if (argc != 12)
    {
        std::fprintf(stderr, "Usage: %s <param> <bin> <embeds> <mask> <cos> <sin> <seq_len> <hidden_ref> <logits_ref> <out_hidden> <out_logits>\n", argv[0]);
        return 2;
    }
    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const int seq_len = std::atoi(argv[7]);

    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    ncnn::Mat embeds = mat_from_f32(argv[3], 1024, seq_len, 1);
    ncnn::Mat mask = mat_from_f32(argv[4], seq_len, seq_len, 1);
    ncnn::Mat cos = rope_mat_from_f32(argv[5], seq_len);
    ncnn::Mat sin = rope_mat_from_f32(argv[6], seq_len);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos);
    ex.input("in3", sin);
    ncnn::Mat hidden, logits;
    if (ex.extract("out0", hidden) != 0 || ex.extract("out1", logits) != 0)
    {
        std::fprintf(stderr, "extract failed\n");
        return 1;
    }

    {
        std::ofstream ofs(argv[10], std::ios::binary);
        ofs.write((const char*)hidden.data, (std::streamsize)(hidden.total() * sizeof(float)));
    }
    {
        std::ofstream ofs(argv[11], std::ios::binary);
        ofs.write((const char*)logits.data, (std::streamsize)(logits.total() * sizeof(float)));
    }
    std::printf("hidden dims w=%d h=%d c=%d total=%zu\n", hidden.w, hidden.h, hidden.c, hidden.total());
    std::printf("logits dims w=%d h=%d c=%d total=%zu\n", logits.w, logits.h, logits.c, logits.total());
    compare("hidden", hidden, argv[8]);
    compare("logits", logits, argv[9]);
    return 0;
}
