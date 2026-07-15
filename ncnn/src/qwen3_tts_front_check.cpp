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

static bool read_f32(const std::string& path, size_t count, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes)) return false;
    if (bytes.size() != count * sizeof(float))
    {
        std::fprintf(stderr, "bad size for %s: %zu expected %zu\n", path.c_str(), bytes.size(), count * sizeof(float));
        return false;
    }
    out.resize(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool write_f32(const std::string& path, const std::vector<float>& x)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)x.data(), (std::streamsize)(x.size() * sizeof(float)));
    return (bool)ofs;
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::fprintf(stderr, "Usage: %s <front_weights_dir> <codes_i32_bin> <code_len> <ref_hidden_f32_bin> <out_hidden_f32_bin> <threads>\n", argv[0]);
        return 2;
    }

    const std::string wdir = argv[1];
    const std::string codes_path = argv[2];
    const int T = std::atoi(argv[3]);
    const std::string ref_path = argv[4];
    const std::string out_path = argv[5];
    const int threads = std::max(1, std::atoi(argv[6]));

    constexpr int Q = 16;
    constexpr int VOCAB = 2048;
    constexpr int EMB = 256;
    constexpr int H = 512;
    constexpr int OUT = 1024;
    constexpr int K = 3;

    std::vector<char> code_bytes;
    if (!read_file(codes_path, code_bytes)) return 1;
    if (code_bytes.size() != (size_t)T * Q * sizeof(int32_t))
    {
        std::fprintf(stderr, "bad code size\n");
        return 1;
    }
    const int32_t* codes = (const int32_t*)code_bytes.data();

    std::vector<float> first_emb;
    std::vector<std::vector<float>> rest_emb(15);
    std::vector<float> first_proj;
    std::vector<float> rest_proj;
    std::vector<float> conv_w;
    std::vector<float> conv_b;

    if (!read_f32(wdir + "/first_embedding.f32.bin", (size_t)VOCAB * EMB, first_emb)) return 1;
    for (int i = 0; i < 15; i++)
        if (!read_f32(wdir + "/rest_embedding_" + std::to_string(i) + ".f32.bin", (size_t)VOCAB * EMB, rest_emb[i])) return 1;
    if (!read_f32(wdir + "/first_output_proj_weight.f32.bin", (size_t)H * EMB, first_proj)) return 1;
    if (!read_f32(wdir + "/rest_output_proj_weight.f32.bin", (size_t)H * EMB, rest_proj)) return 1;
    if (!read_f32(wdir + "/pre_conv_weight.f32.bin", (size_t)OUT * H * K, conv_w)) return 1;
    if (!read_f32(wdir + "/pre_conv_bias.f32.bin", OUT, conv_b)) return 1;

    std::vector<float> quant((size_t)T * H);

#pragma omp parallel for num_threads(threads)
    for (int t = 0; t < T; t++)
    {
        float rest_sum[EMB];
        for (int e = 0; e < EMB; e++) rest_sum[e] = 0.f;

        const int first_id = codes[t * Q + 0];
        const float* first_vec = first_emb.data() + (size_t)first_id * EMB;
        for (int q = 1; q < Q; q++)
        {
            const int id = codes[t * Q + q];
            const float* emb = rest_emb[q - 1].data() + (size_t)id * EMB;
            for (int e = 0; e < EMB; e++)
                rest_sum[e] += emb[e];
        }

        float* qout = quant.data() + (size_t)t * H;
        for (int h = 0; h < H; h++)
        {
            const float* wf = first_proj.data() + (size_t)h * EMB;
            const float* wr = rest_proj.data() + (size_t)h * EMB;
            double sum = 0.0;
            for (int e = 0; e < EMB; e++)
                sum += (double)wf[e] * first_vec[e] + (double)wr[e] * rest_sum[e];
            qout[h] = (float)sum;
        }
    }

    std::vector<float> hidden((size_t)T * OUT);
#pragma omp parallel for num_threads(threads)
    for (int t = 0; t < T; t++)
    {
        for (int oc = 0; oc < OUT; oc++)
        {
            double sum = conv_b[oc];
            for (int ic = 0; ic < H; ic++)
            {
                for (int k = 0; k < K; k++)
                {
                    const int src_t = t + k - 2;
                    if (src_t < 0) continue;
                    const float x = quant[(size_t)src_t * H + ic];
                    const float w = conv_w[((size_t)oc * H + ic) * K + k];
                    sum += (double)x * w;
                }
            }
            hidden[(size_t)t * OUT + oc] = (float)sum;
        }
    }

    if (!write_f32(out_path, hidden)) return 1;

    std::vector<float> ref;
    if (!read_f32(ref_path, (size_t)T * OUT, ref)) return 1;
    double mae = 0.0;
    double rmse = 0.0;
    float maxe = 0.0f;
    for (size_t i = 0; i < hidden.size(); i++)
    {
        const float e = std::fabs(hidden[i] - ref[i]);
        mae += e;
        rmse += (double)e * e;
        maxe = std::max(maxe, e);
    }
    mae /= (double)hidden.size();
    rmse = std::sqrt(rmse / (double)hidden.size());
    std::printf("hidden shape: [%d,%d]\n", T, OUT);
    std::printf("mae=%.9g rmse=%.9g maxe=%.9g\n", mae, rmse, maxe);
    std::printf("wrote %s\n", out_path.c_str());
    return 0;
}
