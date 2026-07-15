#include "net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<char>& data)
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
    if (!read_bytes(path, bytes) || bytes.size() % sizeof(float) != 0) return false;
    data.resize(bytes.size() / sizeof(float));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return true;
}

static bool read_i32(const std::string& path, std::vector<int>& data)
{
    std::vector<char> bytes;
    if (!read_bytes(path, bytes) || bytes.size() % sizeof(int) != 0) return false;
    data.resize(bytes.size() / sizeof(int));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return true;
}

static int argmax_head(const float* hidden, const std::vector<float>& head, int vocab, int dim)
{
    int best = 0;
    float bestv = -3.4e38f;
    for (int v = 0; v < vocab; v++)
    {
        const float* w = head.data() + (size_t)v * dim;
        double acc = 0.0;
        for (int i = 0; i < dim; i++) acc += (double)hidden[i] * w[i];
        const float fv = (float)acc;
        if (fv > bestv)
        {
            bestv = fv;
            best = v;
        }
    }
    return best;
}

static void append_embedding(std::vector<float>& seq, const std::vector<float>& emb, int token, int dim)
{
    const float* row = emb.data() + (size_t)token * dim;
    seq.insert(seq.end(), row, row + dim);
}

static ncnn::Mat make_input_mat(const std::vector<float>& seq, int seq_len, int dim)
{
    ncnn::Mat m(dim, seq_len, (size_t)4u, 1);
    std::memcpy(m.data, seq.data(), (size_t)seq_len * dim * sizeof(float));
    return m;
}

static ncnn::Mat make_mask(int seq_len)
{
    ncnn::Mat m(seq_len, seq_len, 1, (size_t)4u, 1);
    float* p = m;
    const float neg = -3.4028234663852886e38f;
    for (int y = 0; y < seq_len; y++)
    {
        for (int x = 0; x < seq_len; x++)
        {
            p[y * seq_len + x] = x > y ? neg : 0.0f;
        }
    }
    return m;
}

static ncnn::Mat make_pos_f32(int seq_len)
{
    ncnn::Mat m(seq_len, (size_t)4u, 1);
    float* p = (float*)m.data;
    for (int i = 0; i < seq_len; i++) p[i] = (float)i;
    return m;
}

int main(int argc, char** argv)
{
    if (argc != 8)
    {
        std::fprintf(stderr, "Usage: %s <body.param|by_len_dir> <body.bin|bylen> <weights_dir> <hidden_input_f32> <first_code_i32> <tokens_ref_i32> <max_steps>\n", argv[0]);
        return 2;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string weights_dir = argv[3];
    const std::string hidden_path = argv[4];
    const std::string first_code_path = argv[5];
    const std::string tokens_ref_path = argv[6];
    const int max_steps = std::atoi(argv[7]);
    const int dim = 1024;
    const int vocab = 2048;

    std::vector<float> talker_hidden;
    std::vector<int> first_code_vec;
    std::vector<int> ref_tokens;
    if (!read_f32(hidden_path, talker_hidden) || talker_hidden.size() != (size_t)dim ||
        !read_i32(first_code_path, first_code_vec) || first_code_vec.empty() ||
        !read_i32(tokens_ref_path, ref_tokens))
    {
        std::fprintf(stderr, "failed to read inputs\n");
        return 1;
    }

    std::vector<std::vector<float>> heads(15);
    std::vector<std::vector<float>> embs(14);
    std::vector<float> talker_codec_embedding;
    if (!read_f32(weights_dir + "/talker_codec_embedding.f32", talker_codec_embedding) ||
        talker_codec_embedding.size() != (size_t)3072 * dim)
    {
        std::fprintf(stderr, "failed to read talker_codec_embedding\n");
        return 1;
    }
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32(weights_dir + "/lm_head_" + std::to_string(i) + ".f32", heads[i]) ||
            heads[i].size() != (size_t)vocab * dim)
        {
            std::fprintf(stderr, "failed to read lm_head_%d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < 14; i++)
    {
        if (!read_f32(weights_dir + "/codec_embedding_" + std::to_string(i) + ".f32", embs[i]) ||
            embs[i].size() != (size_t)vocab * dim)
        {
            std::fprintf(stderr, "failed to read codec_embedding_%d\n", i);
            return 1;
        }
    }

    const bool by_len = bin_path == "bylen";
    ncnn::Net body_single;
    if (!by_len)
    {
        body_single.opt.num_threads = 4;
        body_single.opt.use_vulkan_compute = false;
        body_single.opt.use_packing_layout = false;
        if (body_single.load_param(param_path.c_str()) != 0 || body_single.load_model(bin_path.c_str()) != 0)
        {
            std::fprintf(stderr, "failed to load body\n");
            return 1;
        }
    }

    std::vector<float> seq = talker_hidden;
    append_embedding(seq, talker_codec_embedding, first_code_vec[0], dim);
    std::vector<int> pred;
    const int steps = std::min(max_steps, 15);
    for (int step = 0; step < steps; step++)
    {
        const int seq_len = (int)seq.size() / dim;
        ncnn::Net body_step;
        ncnn::Net* body = &body_single;
        if (by_len)
        {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "s%02d", seq_len);
            const std::string step_param = param_path + "/code_predictor_body_" + suffix + ".ncnn.param";
            std::string step_bin = param_path + "/code_predictor_body_shared.ncnn.bin";
            {
                std::ifstream shared_test(step_bin, std::ios::binary);
                if (!shared_test) step_bin = param_path + "/code_predictor_body_" + suffix + ".ncnn.bin";
            }
            body_step.opt.num_threads = 4;
            body_step.opt.use_vulkan_compute = false;
            body_step.opt.use_packing_layout = false;
            if (body_step.load_param(step_param.c_str()) != 0 || body_step.load_model(step_bin.c_str()) != 0)
            {
                std::fprintf(stderr, "failed to load %s\n", suffix);
                return 1;
            }
            body = &body_step;
        }
        ncnn::Extractor ex = body->create_extractor();
        ex.input("in0", make_input_mat(seq, seq_len, dim));
        ex.input("in1", make_mask(seq_len));
        ex.input("in2", make_pos_f32(seq_len));
        ncnn::Mat hidden;
        if (ex.extract("out0", hidden) != 0)
        {
            std::fprintf(stderr, "extract failed at step %d seq_len %d\n", step, seq_len);
            return 1;
        }
        const float* hp = (const float*)hidden.data + (size_t)(seq_len - 1) * dim;
        const int token = argmax_head(hp, heads[step], vocab, dim);
        pred.push_back(token);
        std::printf("step=%d seq_len=%d token=%d ref=%d %s\n",
                    step, seq_len, token, step < (int)ref_tokens.size() ? ref_tokens[step] : -1,
                    (step < (int)ref_tokens.size() && token == ref_tokens[step]) ? "OK" : "MISMATCH");
        if (step < steps - 1)
        {
            append_embedding(seq, embs[step], token, dim);
        }
    }
    int mismatches = 0;
    for (int i = 0; i < (int)pred.size() && i < (int)ref_tokens.size(); i++)
    {
        if (pred[i] != ref_tokens[i]) mismatches++;
    }
    std::printf("pred:");
    for (int v : pred) std::printf(" %d", v);
    std::printf("\nmismatches=%d/%zu\n", mismatches, pred.size());
    return mismatches == 0 ? 0 : 3;
}
