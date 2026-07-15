#include "net.h"

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

static bool read_f32_exact(const std::string& path, size_t count, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != count * sizeof(float))
    {
        std::fprintf(stderr, "bad f32 file %s: got %zu expected %zu\n", path.c_str(), bytes.size(), count * sizeof(float));
        return false;
    }
    out.resize(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool read_f32_any(const std::string& path, std::vector<float>& out)
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

static bool read_codes(const std::string& path, int frames, int codebooks, std::vector<int32_t>& codes)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes)) return false;
    const size_t expected = (size_t)frames * codebooks * sizeof(int32_t);
    if (bytes.size() != expected)
    {
        std::fprintf(stderr, "bad code file %s: got %zu expected %zu\n", path.c_str(), bytes.size(), expected);
        return false;
    }
    codes.resize((size_t)frames * codebooks);
    std::memcpy(codes.data(), bytes.data(), bytes.size());
    return true;
}

static std::vector<float> codes_to_hidden(const std::string& wdir, const std::vector<int32_t>& codes, int frames, int threads)
{
    constexpr int Q = 16;
    constexpr int VOCAB = 2048;
    constexpr int EMB = 256;
    constexpr int HIDDEN = 512;
    constexpr int OUT_DIM = 1024;
    constexpr int K = 3;

    std::vector<float> first_emb;
    std::vector<std::vector<float>> rest_emb(15);
    std::vector<float> first_proj;
    std::vector<float> rest_proj;
    std::vector<float> conv_w;
    std::vector<float> conv_b;
    if (!read_f32_exact(wdir + "/first_embedding.f32.bin", (size_t)VOCAB * EMB, first_emb) ||
        !read_f32_exact(wdir + "/first_output_proj_weight.f32.bin", (size_t)HIDDEN * EMB, first_proj) ||
        !read_f32_exact(wdir + "/rest_output_proj_weight.f32.bin", (size_t)HIDDEN * EMB, rest_proj) ||
        !read_f32_exact(wdir + "/pre_conv_weight.f32.bin", (size_t)OUT_DIM * HIDDEN * K, conv_w) ||
        !read_f32_exact(wdir + "/pre_conv_bias.f32.bin", OUT_DIM, conv_b))
    {
        std::exit(1);
    }
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32_exact(wdir + "/rest_embedding_" + std::to_string(i) + ".f32.bin", (size_t)VOCAB * EMB, rest_emb[i]))
            std::exit(1);
    }

    std::vector<float> quant((size_t)frames * HIDDEN);
#pragma omp parallel for num_threads(threads)
    for (int t = 0; t < frames; t++)
    {
        float rest_sum[EMB];
        for (int e = 0; e < EMB; e++) rest_sum[e] = 0.f;
        const int first_id = codes[(size_t)t * Q];
        const float* first_vec = first_emb.data() + (size_t)first_id * EMB;
        for (int q = 1; q < Q; q++)
        {
            const int id = codes[(size_t)t * Q + q];
            const float* emb = rest_emb[q - 1].data() + (size_t)id * EMB;
            for (int e = 0; e < EMB; e++) rest_sum[e] += emb[e];
        }
        float* qout = quant.data() + (size_t)t * HIDDEN;
        for (int h = 0; h < HIDDEN; h++)
        {
            const float* wf = first_proj.data() + (size_t)h * EMB;
            const float* wr = rest_proj.data() + (size_t)h * EMB;
            double sum = 0.0;
            for (int e = 0; e < EMB; e++) sum += (double)wf[e] * first_vec[e] + (double)wr[e] * rest_sum[e];
            qout[h] = (float)sum;
        }
    }

    std::vector<float> hidden((size_t)frames * OUT_DIM);
#pragma omp parallel for num_threads(threads)
    for (int t = 0; t < frames; t++)
    {
        for (int oc = 0; oc < OUT_DIM; oc++)
        {
            double sum = conv_b[oc];
            for (int ic = 0; ic < HIDDEN; ic++)
            {
                for (int k = 0; k < K; k++)
                {
                    const int src_t = t + k - 2;
                    if (src_t < 0) continue;
                    sum += (double)quant[(size_t)src_t * HIDDEN + ic] * conv_w[((size_t)oc * HIDDEN + ic) * K + k];
                }
            }
            hidden[(size_t)t * OUT_DIM + oc] = (float)sum;
        }
    }
    return hidden;
}

static std::vector<float> run_decoder_chunk(ncnn::Net& net, const std::vector<float>& hidden, int hidden_frames, int context_frames, int current_frames, int chunk_frames)
{
    constexpr int DIM = 1024;
    constexpr int SAMPLES_PER_FRAME = 1920;
    std::vector<float> chunk((size_t)chunk_frames * DIM, 0.0f);
    const int copy_frames = context_frames + current_frames;
    if (hidden_frames < copy_frames || copy_frames > chunk_frames)
    {
        std::fprintf(stderr, "bad chunk frames\n");
        std::exit(1);
    }
    std::memcpy(chunk.data(), hidden.data(), (size_t)copy_frames * DIM * sizeof(float));

    ncnn::Mat in(DIM, chunk_frames, (size_t)4u, 1);
    std::memcpy(in.data, chunk.data(), chunk.size() * sizeof(float));
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", in) != 0)
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
    const int drop = context_frames * SAMPLES_PER_FRAME;
    const int keep = current_frames * SAMPLES_PER_FRAME;
    if ((int)out.total() < drop + keep)
    {
        std::fprintf(stderr, "decoder output too small\n");
        std::exit(1);
    }
    const float* p = (const float*)out.data;
    return std::vector<float>(p + drop, p + drop + keep);
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
    if (argc != 13)
    {
        std::fprintf(stderr, "Usage: %s <front_weights_dir> <decoder_param> <decoder_bin> <codes_i32> <frames> <context_frames> <current_frames> <chunk_frames> <ref_wav_f32> <out_hidden_f32> <out_wav_f32> <threads>\n", argv[0]);
        return 2;
    }

    const std::string front_dir = argv[1];
    const std::string decoder_param = argv[2];
    const std::string decoder_bin = argv[3];
    const std::string codes_path = argv[4];
    const int frames = std::atoi(argv[5]);
    const int context_frames = std::atoi(argv[6]);
    const int current_frames = std::atoi(argv[7]);
    const int chunk_frames = std::atoi(argv[8]);
    const std::string ref_path = argv[9];
    const std::string out_hidden_path = argv[10];
    const std::string out_wav_path = argv[11];
    const int threads = std::max(1, std::atoi(argv[12]));

    std::vector<int32_t> codes;
    if (!read_codes(codes_path, frames, 16, codes)) return 1;
    std::vector<float> hidden = codes_to_hidden(front_dir, codes, frames, threads);
    if (!write_f32(out_hidden_path, hidden)) return 1;

    ncnn::Net decoder;
    decoder.opt.num_threads = threads;
    decoder.opt.use_vulkan_compute = false;
    decoder.opt.use_packing_layout = false;
    if (decoder.load_param(decoder_param.c_str()) != 0 || decoder.load_model(decoder_bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load decoder\n");
        return 1;
    }
    std::vector<float> wav = run_decoder_chunk(decoder, hidden, frames, context_frames, current_frames, chunk_frames);
    if (!write_f32(out_wav_path, wav)) return 1;
    std::printf("hidden_frames=%d wav_samples=%zu wrote=%s\n", frames, wav.size(), out_wav_path.c_str());

    std::vector<float> ref;
    if (read_f32_any(ref_path, ref)) compare(wav, ref);
    return 0;
}
