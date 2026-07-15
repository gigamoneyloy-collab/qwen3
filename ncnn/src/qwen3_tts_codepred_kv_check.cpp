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

static ncnn::Mat mat_from_f32(const std::string& path, int w, int h)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != (size_t)w * h * sizeof(float))
    {
        std::fprintf(stderr, "bad f32 file %s\n", path.c_str());
        return ncnn::Mat();
    }
    ncnn::Mat m(w, h, (size_t)4u, 1);
    std::memcpy(m.data, bytes.data(), bytes.size());
    return m;
}

static int argmax(const ncnn::Mat& m)
{
    const float* p = (const float*)m.data;
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < (int)m.total(); i++)
    {
        if (p[i] > bestv)
        {
            bestv = p[i];
            best = i;
        }
    }
    return best;
}

static int compare_ref(const ncnn::Mat& out, const std::string& ref_path)
{
    std::vector<char> ref_bytes;
    if (!read_file(ref_path, ref_bytes) || ref_bytes.size() % sizeof(float) != 0)
    {
        std::fprintf(stderr, "bad ref file %s\n", ref_path.c_str());
        return 1;
    }
    const float* ref = (const float*)ref_bytes.data();
    const size_t ref_count = ref_bytes.size() / sizeof(float);
    const float* got = (const float*)out.data;
    const size_t n = std::min(ref_count, out.total());
    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.f;
    for (size_t i = 0; i < n; i++)
    {
        const float e = std::fabs(got[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
    }
    if (n)
    {
        mae /= (double)n;
        rmse = std::sqrt(rmse / (double)n);
    }
    std::printf("ref_count=%zu compare_count=%zu mae=%.9g rmse=%.9g maxe=%.9g\n", ref_count, n, mae, rmse, maxe);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 9 && argc != 15)
    {
        std::fprintf(stderr,
                     "Usage:\n"
                     "  %s <prefill_param> <prefill_bin> <input_f32> <mask_f32> <pos_f32_or_i64> <ref_logits_f32> <threads> <use_vulkan>\n"
                     "  %s <prefill_param> <prefill_bin> <input_f32> <mask_f32> <pos_f32_or_i64> <ref_logits_f32> <threads> <use_vulkan> <decode_param> <decode_bin> <decode_input_f32> <decode_mask_f32> <decode_ref_logits_f32> <decode_expected_argmax>\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::string param = argv[1];
    const std::string bin = argv[2];
    const std::string input_path = argv[3];
    const std::string mask_path = argv[4];
    const std::string pos_path = argv[5];
    const std::string ref_logits = argv[6];
    const int threads = std::max(1, std::atoi(argv[7]));
    const bool use_vulkan = std::atoi(argv[8]) != 0;

    auto configure = [&](ncnn::Net& net) {
        net.opt.num_threads = threads;
        net.opt.use_vulkan_compute = use_vulkan;
        net.opt.use_packing_layout = false;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
    };

    ncnn::Net net;
    configure(net);
    if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    ncnn::Mat in0 = mat_from_f32(input_path, 1024, 2);
    ncnn::Mat in1 = mat_from_f32(mask_path, 2, 2);
    ncnn::Mat in2(2, (size_t)4u, 1);
    float* pp = in2;
    pp[0] = 0.f;
    pp[1] = 1.f;
    (void)pos_path;
    if (in0.empty() || in1.empty() || in2.empty()) return 1;

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    if (ex.input("in0", in0) != 0 || ex.input("in1", in1) != 0 || ex.input("in2", in2) != 0)
    {
        std::fprintf(stderr, "failed to set inputs\n");
        return 1;
    }

    ncnn::Mat hidden;
    ncnn::Mat logits;
    if (ex.extract("out0", hidden) != 0 || ex.extract("out1", logits) != 0)
    {
        std::fprintf(stderr, "failed to extract out0/out1\n");
        return 1;
    }
    std::printf("hidden w=%d h=%d c=%d total=%zu\n", hidden.w, hidden.h, hidden.c, hidden.total());
    std::printf("logits w=%d h=%d c=%d total=%zu argmax=%d\n", logits.w, logits.h, logits.c, logits.total(), argmax(logits));
    compare_ref(logits, ref_logits);

    std::vector<ncnn::Mat> cache_k(5), cache_v(5);
    for (int i = 0; i < 5; i++)
    {
        char kname[32], vname[32];
        std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
        std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
        if (ex.extract(kname, cache_k[i]) != 0 || ex.extract(vname, cache_v[i]) != 0)
        {
            std::fprintf(stderr, "failed to extract cache %d\n", i);
            return 1;
        }
        std::printf("cache%d k=(%d,%d,%d) total=%zu v=(%d,%d,%d) total=%zu\n",
                    i, cache_k[i].w, cache_k[i].h, cache_k[i].c, cache_k[i].total(), cache_v[i].w, cache_v[i].h, cache_v[i].c, cache_v[i].total());
    }

    if (argc == 15)
    {
        const std::string decode_param = argv[9];
        const std::string decode_bin = argv[10];
        const std::string decode_input_path = argv[11];
        const std::string decode_mask_path = argv[12];
        const std::string decode_ref_logits = argv[13];
        const int expected_argmax = std::atoi(argv[14]);

        ncnn::Net decode_net;
        configure(decode_net);
        if (decode_net.load_param(decode_param.c_str()) != 0 || decode_net.load_model(decode_bin.c_str()) != 0)
        {
            std::fprintf(stderr, "failed to load decode model\n");
            return 1;
        }

        ncnn::Mat din0 = mat_from_f32(decode_input_path, 1024, 1);
        ncnn::Mat din1 = mat_from_f32(decode_mask_path, 3, 1);
        ncnn::Mat din2(1, (size_t)4u, 1);
        ((float*)din2.data)[0] = 2.f;
        if (din0.empty() || din1.empty() || din2.empty()) return 1;

        ncnn::Extractor dex = decode_net.create_extractor();
        dex.set_light_mode(false);
        if (dex.input("in0", din0) != 0 || dex.input("in1", din1) != 0 || dex.input("in2", din2) != 0)
        {
            std::fprintf(stderr, "failed to set decode inputs\n");
            return 1;
        }
        for (int i = 0; i < 5; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            if (dex.input(kname, cache_k[i]) != 0 || dex.input(vname, cache_v[i]) != 0)
            {
                std::fprintf(stderr, "failed to set decode cache %d\n", i);
                return 1;
            }
        }
        ncnn::Mat dhidden, dlogits;
        if (dex.extract("out0", dhidden) != 0 || dex.extract("out1", dlogits) != 0)
        {
            std::fprintf(stderr, "failed to extract decode out0/out1\n");
            return 1;
        }
        const int got_argmax = argmax(dlogits);
        std::printf("decode hidden w=%d h=%d c=%d total=%zu\n", dhidden.w, dhidden.h, dhidden.c, dhidden.total());
        std::printf("decode logits w=%d h=%d c=%d total=%zu argmax=%d expected=%d\n",
                    dlogits.w, dlogits.h, dlogits.c, dlogits.total(), got_argmax, expected_argmax);
        compare_ref(dlogits, decode_ref_logits);
        if (got_argmax != expected_argmax) return 1;
    }
    return 0;
}
