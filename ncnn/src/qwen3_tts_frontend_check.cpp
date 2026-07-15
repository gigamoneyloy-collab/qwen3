#include "qwen3_tts_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

static bool read_text_arg(const std::string& arg, std::string& out)
{
    if (arg.empty() || arg[0] != '@')
    {
        out = arg;
        return true;
    }
    std::ifstream ifs(arg.substr(1), std::ios::binary);
    if (!ifs) return false;
    out.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        std::fprintf(stderr, "Usage: %s <tokenizer.txt> <text_embed.param> <text_embed.bin> <codec_embed.param> <codec_embed.bin> <text> <language> <speaker> <ref_prompt_f32>\n", argv[0]);
        return 2;
    }

    Qwen3TTSFrontendFiles files;
    files.tokenizer = argv[1];
    files.text_embed_param = argv[2];
    files.text_embed_bin = argv[3];
    files.codec_embed_param = argv[4];
    files.codec_embed_bin = argv[5];

    Qwen3TTSFrontend frontend;
    if (!frontend.load(files, 4, false)) return 1;
    std::string text;
    if (!read_text_arg(argv[6], text))
    {
        std::fprintf(stderr, "failed to read text %s\n", argv[6]);
        return 1;
    }
    ncnn::Mat prompt = frontend.build_customvoice_prompt(text, argv[7], argv[8]);
    if (prompt.empty()) return 1;

    std::vector<float> ref;
    if (!read_f32(argv[9], ref))
    {
        std::fprintf(stderr, "failed to read ref %s\n", argv[9]);
        return 1;
    }
    const size_t got_n = prompt.total();
    const float* got = (const float*)prompt.data;
    const size_t n = std::min(got_n, ref.size());
    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.0f;
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
    std::printf("prompt_shape=(%d,%d) got=%zu ref=%zu compare=%zu\n", prompt.h, prompt.w, got_n, ref.size(), n);
    std::printf("prompt mae=%.9g rmse=%.9g maxe=%.9g\n", mae, rmse, maxe);
    return got_n == ref.size() && maxe < 1e-4f ? 0 : 1;
}
