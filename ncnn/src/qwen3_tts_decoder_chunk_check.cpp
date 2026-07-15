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

static bool read_f32(const std::string& path, std::vector<float>& data)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(float) != 0) return false;
    data.resize(bytes.size() / sizeof(float));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return true;
}

static bool write_f32(const std::string& path, const std::vector<float>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    return (bool)ofs;
}

static ncnn::Mat mat_from_hidden(const std::vector<float>& hidden, int frames, int dim)
{
    ncnn::Mat m(dim, frames, (size_t)4u, 1);
    std::memcpy(m.data, hidden.data(), (size_t)frames * dim * sizeof(float));
    return m;
}

static std::vector<float> run_decoder(ncnn::Net& net, const std::vector<float>& hidden, int total_frames, int dim)
{
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", mat_from_hidden(hidden, total_frames, dim)) != 0)
    {
        std::fprintf(stderr, "failed to set decoder input\n");
        std::exit(1);
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)
    {
        std::fprintf(stderr, "failed to extract decoder output\n");
        std::exit(1);
    }
    std::vector<float> wav(out.total());
    std::copy((const float*)out.data, (const float*)out.data + out.total(), wav.begin());
    return wav;
}

static void compare_prefix(const std::vector<float>& got, const std::vector<float>& ref)
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
    if (argc != 12)
    {
        std::fprintf(stderr,
                     "Usage: %s <param> <bin> <hidden_f32> <hidden_frames> <context_frames> <current_frames> <chunk_frames> <hidden_dim> <ref_wav_f32> <out_wav_f32> <threads>\n",
                     argv[0]);
        return 2;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string hidden_path = argv[3];
    const int hidden_frames = std::atoi(argv[4]);
    const int context_frames = std::atoi(argv[5]);
    const int current_frames = std::atoi(argv[6]);
    const int chunk_frames = std::atoi(argv[7]);
    const int dim = std::atoi(argv[8]);
    const std::string ref_path = argv[9];
    const std::string out_path = argv[10];
    const int threads = std::max(1, std::atoi(argv[11]));
    const int samples_per_frame = 1920;

    if (context_frames < 0 || current_frames <= 0 || chunk_frames <= 0 || dim <= 0)
    {
        std::fprintf(stderr, "bad frame/dim arguments\n");
        return 2;
    }
    if (context_frames + current_frames > chunk_frames)
    {
        std::fprintf(stderr, "context + current exceeds chunk_frames\n");
        return 2;
    }
    if (hidden_frames < context_frames + current_frames)
    {
        std::fprintf(stderr, "hidden file does not contain context + current frames\n");
        return 2;
    }

    std::vector<float> hidden_in;
    if (!read_f32(hidden_path, hidden_in))
    {
        std::fprintf(stderr, "failed to read %s\n", hidden_path.c_str());
        return 1;
    }
    if (hidden_in.size() != (size_t)hidden_frames * dim)
    {
        std::fprintf(stderr, "unexpected hidden size %zu, expected %zu\n", hidden_in.size(), (size_t)hidden_frames * dim);
        return 1;
    }

    std::vector<float> chunk((size_t)chunk_frames * dim, 0.0f);
    const int copy_frames = context_frames + current_frames;
    std::memcpy(chunk.data(), hidden_in.data(), (size_t)copy_frames * dim * sizeof(float));

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load decoder graph\n");
        return 1;
    }

    std::vector<float> full_wav = run_decoder(net, chunk, chunk_frames, dim);
    const int drop_samples = context_frames * samples_per_frame;
    const int keep_samples = current_frames * samples_per_frame;
    if ((int)full_wav.size() < drop_samples + keep_samples)
    {
        std::fprintf(stderr, "decoder output too small: %zu\n", full_wav.size());
        return 1;
    }
    std::vector<float> cropped(full_wav.begin() + drop_samples, full_wav.begin() + drop_samples + keep_samples);
    if (!write_f32(out_path, cropped)) return 1;

    std::printf("decoder_out=%zu drop=%d keep=%d wrote=%s\n", full_wav.size(), drop_samples, keep_samples, out_path.c_str());
    std::vector<float> ref;
    if (read_f32(ref_path, ref))
        compare_prefix(cropped, ref);
    else
        std::fprintf(stderr, "warning: failed to read ref %s\n", ref_path.c_str());

    return 0;
}
