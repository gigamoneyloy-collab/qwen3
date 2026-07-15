#include "qwen3_tts_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct ModelJson {
    std::string front_weights_dir;
    std::string decoder_param;
    std::string decoder_bin;
    std::string talker_prefill_param;
    std::string talker_decode_param;
    std::string talker_bin;
    std::string codepred_body_dir;
    std::string codepred_kv_dir;
    std::string codepred_weights_dir;
    std::string tts_pad_embed;
    std::string prefill_embed;
    std::string prefill_mask;
    std::string prefill_cos;
    std::string prefill_sin;
    int decoder_chunk_frames = 325;
    int decoder_context_frames = 25;
};

static bool read_text(const std::string& path, std::string& out)
{
    std::ifstream ifs(path);
    if (!ifs) return false;
    out.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return true;
}

static std::string json_string(const std::string& s, const std::string& key)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return {};
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return {};
    p = s.find('"', p + 1);
    if (p == std::string::npos) return {};
    size_t q = s.find('"', p + 1);
    if (q == std::string::npos) return {};
    return s.substr(p + 1, q - p - 1);
}

static int json_int(const std::string& s, const std::string& key, int fallback)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return fallback;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return fallback;
    return std::atoi(s.c_str() + p + 1);
}

static bool load_model_json(const std::string& path, ModelJson& cfg)
{
    std::string s;
    if (!read_text(path, s)) return false;
    cfg.front_weights_dir = json_string(s, "front_weights_dir");
    cfg.decoder_param = json_string(s, "decoder_param");
    cfg.decoder_bin = json_string(s, "decoder_bin");
    cfg.talker_prefill_param = json_string(s, "talker_prefill_param");
    cfg.talker_decode_param = json_string(s, "talker_decode_param");
    cfg.talker_bin = json_string(s, "talker_bin");
    cfg.codepred_body_dir = json_string(s, "codepred_body_dir");
    cfg.codepred_kv_dir = json_string(s, "codepred_kv_dir");
    cfg.codepred_weights_dir = json_string(s, "codepred_weights_dir");
    cfg.tts_pad_embed = json_string(s, "tts_pad_embed");
    cfg.prefill_embed = json_string(s, "prefill_embed");
    cfg.prefill_mask = json_string(s, "prefill_mask");
    cfg.prefill_cos = json_string(s, "prefill_cos");
    cfg.prefill_sin = json_string(s, "prefill_sin");
    cfg.decoder_chunk_frames = json_int(s, "decoder_chunk_frames", 325);
    cfg.decoder_context_frames = json_int(s, "decoder_context_frames", 25);
    return !cfg.front_weights_dir.empty() && !cfg.decoder_param.empty() && !cfg.talker_prefill_param.empty();
}

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

static bool read_i32(const std::string& path, std::vector<int32_t>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(int32_t) != 0) return false;
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool write_i32(const std::string& path, const std::vector<int32_t>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(int32_t)));
    return (bool)ofs;
}

static bool write_f32(const std::string& path, const std::vector<float>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    return (bool)ofs;
}

static int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    return std::atoi(v);
}

static void compare_wav(const std::vector<float>& got, const std::vector<float>& ref)
{
    const size_t n = std::min(got.size(), ref.size());
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
    std::printf("wav ref_count=%zu compare_count=%zu\n", ref.size(), n);
    std::printf("wav mae=%.9g rmse=%.9g maxe=%.9g\n", mae, rmse, maxe);
}

int main(int argc, char** argv)
{
    if (argc != 9)
    {
        std::fprintf(stderr, "Usage: %s <model.json> <frames> <ref_codes_i32> <ref_wav_f32> <out_codes_i32> <out_wav_f32> <threads> <use_vulkan>\n", argv[0]);
        return 2;
    }

    ModelJson cfg;
    if (!load_model_json(argv[1], cfg))
    {
        std::fprintf(stderr, "failed to load model json %s\n", argv[1]);
        return 1;
    }

    Qwen3TTSNcnn::Options opt;
    opt.num_threads = std::max(1, std::atoi(argv[7]));
    opt.decoder_num_threads = env_int("QWEN3_TTS_DECODER_THREADS", 0);
    opt.use_vulkan = std::atoi(argv[8]) != 0;
    opt.codepred_use_vulkan = opt.use_vulkan;
    if (env_int("QWEN3_TTS_CODEPRED_CPU", 0) > 0) opt.codepred_use_vulkan = false;
    opt.codepred_use_fp16 = env_int("QWEN3_TTS_CODEPRED_FP16", 0) > 0;
    opt.codepred_fp16_mode = env_int("QWEN3_TTS_CODEPRED_FP16_MODE", 0);
    opt.decoder_chunk_frames = cfg.decoder_chunk_frames;
    opt.decoder_context_frames = cfg.decoder_context_frames;

    Qwen3TTSNcnn tts(cfg.front_weights_dir, cfg.decoder_param, cfg.decoder_bin, opt);
    if (!tts.ok()) return 1;

    Qwen3TTSNcnn::TalkerFiles talker;
    talker.prefill_param = cfg.talker_prefill_param;
    talker.decode_param = cfg.talker_decode_param;
    talker.talker_bin = cfg.talker_bin;
    talker.codepred_body_dir = cfg.codepred_body_dir;
    talker.codepred_kv_dir = cfg.codepred_kv_dir;
    talker.codepred_weights_dir = cfg.codepred_weights_dir;
    talker.tts_pad_embed = cfg.tts_pad_embed;
    if (!tts.load_talker(talker)) return 1;

    const int frames = std::atoi(argv[2]);
    std::vector<int32_t> codes = tts.generate_codes(cfg.prefill_embed, cfg.prefill_mask, cfg.prefill_cos, cfg.prefill_sin, frames);
    if (codes.empty() || !write_i32(argv[5], codes)) return 1;

    std::vector<int32_t> ref_codes;
    if (read_i32(argv[3], ref_codes))
    {
        int mismatch = 0;
        const size_t n = std::min(codes.size(), ref_codes.size());
        for (size_t i = 0; i < n; i++) if (codes[i] != ref_codes[i]) mismatch++;
        std::printf("code mismatches=%d/%zu generated=%zu ref=%zu\n", mismatch, n, codes.size(), ref_codes.size());
    }

    std::vector<float> wav = tts.codes_to_wav(codes, frames, 0, frames);
    if (wav.empty() || !write_f32(argv[6], wav)) return 1;
    std::printf("wav_samples=%zu wrote=%s\n", wav.size(), argv[6]);

    std::vector<float> ref_wav;
    if (read_f32(argv[4], ref_wav)) compare_wav(wav, ref_wav);
    return 0;
}
