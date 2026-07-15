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

static bool read_codes(const std::string& path, int frames, std::vector<int32_t>& codes)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes)) return false;
    const size_t expected = (size_t)frames * Qwen3TTSNcnn::codebooks * sizeof(int32_t);
    if (bytes.size() != expected) return false;
    codes.resize((size_t)frames * Qwen3TTSNcnn::codebooks);
    std::memcpy(codes.data(), bytes.data(), bytes.size());
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

static bool write_f32(const std::string& path, const std::vector<float>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    return (bool)ofs;
}

static void compare(const std::vector<float>& got, const std::vector<float>& ref)
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
    std::printf("ref_count=%zu compare_count=%zu\n", ref.size(), n);
    std::printf("mae=%.9g rmse=%.9g maxe=%.9g\n", mae, rmse, maxe);
}

int main(int argc, char** argv)
{
    if (argc != 12 && argc != 13)
    {
        std::fprintf(stderr, "Usage: %s <front_weights_dir> <decoder_param> <decoder_bin> <codes_i32> <frames> <context_frames> <current_frames> <ref_wav_f32> <out_wav_f32> <threads> <chunk_frames> [use_vulkan]\n", argv[0]);
        return 2;
    }

    const std::string front_dir = argv[1];
    const std::string decoder_param = argv[2];
    const std::string decoder_bin = argv[3];
    const std::string codes_path = argv[4];
    const int frames = std::atoi(argv[5]);
    const int context_frames = std::atoi(argv[6]);
    const int current_frames = std::atoi(argv[7]);
    const std::string ref_path = argv[8];
    const std::string out_path = argv[9];
    const int threads = std::max(1, std::atoi(argv[10]));
    const int chunk_frames = std::atoi(argv[11]);
    const bool use_vulkan = argc == 13 && std::atoi(argv[12]) != 0;

    std::vector<int32_t> codes;
    if (!read_codes(codes_path, frames, codes))
    {
        std::fprintf(stderr, "failed to read codes\n");
        return 1;
    }

    Qwen3TTSNcnn::Options opt;
    opt.num_threads = threads;
    opt.decoder_chunk_frames = chunk_frames;
    opt.use_vulkan = use_vulkan;
    Qwen3TTSNcnn tts(front_dir, decoder_param, decoder_bin, opt);
    if (!tts.ok()) return 1;

    std::vector<float> wav = tts.codes_to_wav(codes, frames, context_frames, current_frames);
    if (wav.empty() || !write_f32(out_path, wav)) return 1;
    std::printf("wav_samples=%zu wrote=%s\n", wav.size(), out_path.c_str());

    std::vector<float> ref;
    if (read_f32(ref_path, ref)) compare(wav, ref);
    return 0;
}
