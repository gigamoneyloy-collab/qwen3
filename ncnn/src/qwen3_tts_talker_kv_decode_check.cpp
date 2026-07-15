#include "net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
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

static ncnn::Mat read_mat2(const std::string& path, int w, int h)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != (size_t)w * h * sizeof(float))
    {
        std::fprintf(stderr, "bad file %s size=%zu expect=%zu\n", path.c_str(), bytes.size(), (size_t)w * h * sizeof(float));
        std::exit(1);
    }
    ncnn::Mat m(w, h, (size_t)4u, 1);
    std::memcpy(m.data, bytes.data(), bytes.size());
    return m;
}

static void compare_last_logits(const ncnn::Mat& logits, const std::string& ref_path, int ref_seq_len)
{
    std::vector<char> bytes;
    if (!read_file(ref_path, bytes)) std::exit(1);
    const float* ref = (const float*)bytes.data() + (size_t)(ref_seq_len - 1) * 3072;
    const float* out = (const float*)logits.data;
    double mae = 0.0, rmse = 0.0;
    float maxe = 0.0f;
    int argmax_out = 0, argmax_ref = 0;
    float best_out = out[0], best_ref = ref[0];
    for (int i = 0; i < 3072; i++)
    {
        const float e = std::fabs(out[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
        if (out[i] > best_out) { best_out = out[i]; argmax_out = i; }
        if (ref[i] > best_ref) { best_ref = ref[i]; argmax_ref = i; }
    }
    mae /= 3072.0;
    rmse = std::sqrt(rmse / 3072.0);
    std::printf("decode_last_logits mae=%.9g rmse=%.9g maxe=%.9g argmax_out=%d argmax_ref=%d\n",
                mae, rmse, maxe, argmax_out, argmax_ref);
}

int main(int argc, char** argv)
{
    if (argc != 13)
    {
        std::fprintf(stderr, "Usage: %s <prefill_param> <decode_param> <bin> <prefill_embed> <prefill_mask> <prefill_cos> <prefill_sin> <decode_embed> <decode_mask> <decode_cos> <decode_sin> <full23_logits_ref>\n", argv[0]);
        return 2;
    }

    ncnn::Net prefill_net;
    prefill_net.opt.num_threads = 4;
    prefill_net.opt.use_vulkan_compute = false;
    prefill_net.opt.use_packing_layout = false;
    if (prefill_net.load_param(argv[1]) != 0 || prefill_net.load_model(argv[3]) != 0)
    {
        std::fprintf(stderr, "failed to load prefill model\n");
        return 1;
    }
    ncnn::Net decode_net;
    decode_net.opt.num_threads = 4;
    decode_net.opt.use_vulkan_compute = false;
    decode_net.opt.use_packing_layout = false;
    if (decode_net.load_param(argv[2]) != 0 || decode_net.load_model(argv[3]) != 0)
    {
        std::fprintf(stderr, "failed to load decode model\n");
        return 1;
    }

    const int prefill_len = 22;
    const int decode_len = 1;
    ncnn::Mat prefill_embed = read_mat2(argv[4], 1024, prefill_len);
    ncnn::Mat prefill_mask = read_mat2(argv[5], prefill_len, prefill_len);
    ncnn::Mat prefill_cos = read_mat2(argv[6], 128, prefill_len);
    ncnn::Mat prefill_sin = read_mat2(argv[7], 128, prefill_len);

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> cache;
    {
        ncnn::Extractor ex = prefill_net.create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", prefill_embed);
        ex.input("in1", prefill_mask);
        ex.input("in2", prefill_cos);
        ex.input("in3", prefill_sin);
        for (int i = 0; i < 28; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k, v;
            if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0)
            {
                std::fprintf(stderr, "prefill cache extract failed %d\n", i);
                return 1;
            }
            cache.emplace_back(std::move(k), std::move(v));
        }
        ncnn::Mat logits;
        ex.extract("out1", logits);
        std::printf("prefill caches=%zu cache0_k=(w=%d,h=%d,c=%d,total=%zu)\n",
                    cache.size(), cache[0].first.w, cache[0].first.h, cache[0].first.c, cache[0].first.total());
    }

    ncnn::Mat decode_embed = read_mat2(argv[8], 1024, decode_len);
    ncnn::Mat decode_mask = read_mat2(argv[9], prefill_len + 1, 1);
    ncnn::Mat decode_cos = read_mat2(argv[10], 128, decode_len);
    ncnn::Mat decode_sin = read_mat2(argv[11], 128, decode_len);
    ncnn::Mat decode_logits;
    {
        ncnn::Extractor ex = decode_net.create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", decode_embed);
        ex.input("in1", decode_mask);
        ex.input("in2", decode_cos);
        ex.input("in3", decode_sin);
        for (int i = 0; i < 28; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, cache[i].first);
            ex.input(vname, cache[i].second);
        }
        if (ex.extract("out1", decode_logits) != 0)
        {
            std::fprintf(stderr, "decode logits extract failed\n");
            return 1;
        }
    }
    std::printf("decode logits dims w=%d h=%d c=%d total=%zu\n", decode_logits.w, decode_logits.h, decode_logits.c, decode_logits.total());
    compare_last_logits(decode_logits, argv[12], 23);
    return 0;
}
