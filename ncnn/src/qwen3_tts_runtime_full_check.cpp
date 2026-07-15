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

static bool read_i32(const std::string& path, std::vector<int32_t>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(int32_t) != 0) return false;
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool read_f32(const std::string& path, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(float) != 0) return false;
    out.resize(bytes.size() / sizeof(float));
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

template<typename T>
static int count_mismatch(const std::vector<T>& a, const std::vector<T>& b)
{
    const size_t n = std::min(a.size(), b.size());
    int m = 0;
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) m++;
    return m;
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
    if (argc != 24)
    {
        std::fprintf(stderr, "Usage: %s <front_weights_dir> <decoder_param> <decoder_bin> <talker_prefill_param> <talker_decode_param> <talker_bin> <codepred_body_dir> <codepred_weights_dir> <tts_pad_embed> <prefill_embed> <prefill_mask> <prefill_cos> <prefill_sin> <frames> <ref_codes_i32> <ref_wav_f32> <out_codes_i32> <out_wav_f32> <threads> <chunk_frames> <context_frames> <current_frames> <use_vulkan>\n", argv[0]);
        return 2;
    }

    Qwen3TTSNcnn::Options opt;
    opt.num_threads = std::max(1, std::atoi(argv[19]));
    opt.decoder_chunk_frames = std::atoi(argv[20]);
    opt.decoder_context_frames = std::atoi(argv[21]);
    opt.use_vulkan = std::atoi(argv[23]) != 0;

    Qwen3TTSNcnn tts(argv[1], argv[2], argv[3], opt);
    if (!tts.ok()) return 1;

    Qwen3TTSNcnn::TalkerFiles talker;
    talker.prefill_param = argv[4];
    talker.decode_param = argv[5];
    talker.talker_bin = argv[6];
    talker.codepred_body_dir = argv[7];
    talker.codepred_weights_dir = argv[8];
    talker.tts_pad_embed = argv[9];
    if (!tts.load_talker(talker)) return 1;

    const int frames = std::atoi(argv[14]);
    const int current_frames = std::atoi(argv[22]);
    std::vector<int32_t> codes = tts.generate_codes(argv[10], argv[11], argv[12], argv[13], frames);
    if (codes.empty() || !write_i32(argv[17], codes)) return 1;

    std::vector<int32_t> ref_codes;
    if (read_i32(argv[15], ref_codes))
    {
        std::printf("code mismatches=%d/%zu generated=%zu ref=%zu\n",
                    count_mismatch(codes, ref_codes),
                    std::min(codes.size(), ref_codes.size()),
                    codes.size(),
                    ref_codes.size());
    }

    std::vector<float> wav = tts.codes_to_wav(codes, frames, opt.decoder_context_frames, current_frames);
    if (wav.empty() || !write_f32(argv[18], wav)) return 1;
    std::printf("wav_samples=%zu wrote=%s\n", wav.size(), argv[18]);

    std::vector<float> ref_wav;
    if (read_f32(argv[16], ref_wav)) compare_wav(wav, ref_wav);
    return 0;
}
