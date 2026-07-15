#include "qwen3_tts_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

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

static int env_int(const char* name, int fallback)
{
    const char* p = std::getenv(name);
    if (!p || !p[0]) return fallback;
    const int v = std::atoi(p);
    return v > 0 ? v : fallback;
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

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void print_stats(const char* name, const std::vector<double>& v)
{
    if (v.empty()) return;
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    const auto mm = std::minmax_element(v.begin(), v.end());
    std::printf("stat,%s,mean_ms,%.3f,median_ms,%.3f,min_ms,%.3f,max_ms,%.3f\n",
                name, sum / (double)v.size(), median(v), *mm.first, *mm.second);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::fprintf(stderr, "Usage: %s <model.json> <frames> <threads> <use_vulkan 0|1> <warmup> <repeat>\n", argv[0]);
        return 2;
    }

    const std::string model_json = argv[1];
    const int frames = std::atoi(argv[2]);
    const int threads = std::max(1, std::atoi(argv[3]));
    const bool use_vulkan = std::atoi(argv[4]) != 0;
    const int warmup = std::max(0, std::atoi(argv[5]));
    const int repeat = std::max(1, std::atoi(argv[6]));

    ModelJson cfg;
    if (!load_model_json(model_json, cfg))
    {
        std::fprintf(stderr, "failed to load model json %s\n", model_json.c_str());
        return 1;
    }

    Qwen3TTSNcnn::Options opt;
    opt.num_threads = threads;
    opt.decoder_num_threads = env_int("QWEN3_TTS_DECODER_THREADS", 0);
    opt.use_vulkan = use_vulkan;
    opt.codepred_use_vulkan = use_vulkan;
    if (env_int("QWEN3_TTS_CODEPRED_CPU", 0) > 0) opt.codepred_use_vulkan = false;
    opt.codepred_use_fp16 = env_int("QWEN3_TTS_CODEPRED_FP16", 0) > 0;
    opt.codepred_fp16_mode = env_int("QWEN3_TTS_CODEPRED_FP16_MODE", 0);
    opt.decoder_chunk_frames = cfg.decoder_chunk_frames;
    opt.decoder_context_frames = cfg.decoder_context_frames;

    const double load_t0 = now_ms();
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
    const double load_ms = now_ms() - load_t0;
    std::printf("load_ms,%.3f\n", load_ms);
    std::printf("iter,phase,generate_ms,wav_ms,total_ms,codes,wav_samples\n");

    std::vector<double> gen_times;
    std::vector<double> wav_times;
    std::vector<double> total_times;
    for (int iter = -warmup; iter < repeat; iter++)
    {
        const double total_t0 = now_ms();
        const double gen_t0 = now_ms();
        std::vector<int32_t> codes = tts.generate_codes(cfg.prefill_embed, cfg.prefill_mask, cfg.prefill_cos, cfg.prefill_sin, frames);
        const double gen_ms = now_ms() - gen_t0;
        const double wav_t0 = now_ms();
        std::vector<float> wav = tts.codes_to_wav(codes, frames, 0, frames);
        const double wav_ms = now_ms() - wav_t0;
        const double total_ms = now_ms() - total_t0;
        if (codes.empty() || wav.empty()) return 1;
        const char* phase = iter < 0 ? "warmup" : "repeat";
        std::printf("%d,%s,%.3f,%.3f,%.3f,%zu,%zu\n",
                    iter, phase, gen_ms, wav_ms, total_ms, codes.size(), wav.size());
        if (iter >= 0)
        {
            gen_times.push_back(gen_ms);
            wav_times.push_back(wav_ms);
            total_times.push_back(total_ms);
        }
    }

    print_stats("generate", gen_times);
    print_stats("wav", wav_times);
    print_stats("total", total_times);
    return 0;
}
