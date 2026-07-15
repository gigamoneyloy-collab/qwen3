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

static ncnn::Mat mat2_from_vec(const std::vector<float>& v, int w, int h)
{
    ncnn::Mat m(w, h, (size_t)4u, 1);
    std::memcpy(m.data, v.data(), (size_t)w * h * sizeof(float));
    return m;
}

static ncnn::Mat mat2_from_file(const std::string& path, int w, int h)
{
    std::vector<float> v;
    if (!read_f32(path, v) || v.size() != (size_t)w * h)
    {
        std::fprintf(stderr, "bad f32 file %s\n", path.c_str());
        std::exit(1);
    }
    return mat2_from_vec(v, w, h);
}

static ncnn::Mat make_decode_mask(int kv_len)
{
    ncnn::Mat m(kv_len + 1, 1, (size_t)4u, 1);
    m.fill(0.0f);
    return m;
}

static void make_rope(int position, ncnn::Mat& cos, ncnn::Mat& sin)
{
    const int dim = 128;
    const int half = 64;
    cos.create(dim, 1, (size_t)4u, 1);
    sin.create(dim, 1, (size_t)4u, 1);
    float* cp = cos;
    float* sp = sin;
    for (int i = 0; i < half; i++)
    {
        const float inv = std::pow(1000000.0f, -2.0f * i / (float)dim);
        const float a = position * inv;
        const float c = std::cos(a);
        const float s = std::sin(a);
        cp[i] = c;
        cp[i + half] = c;
        sp[i] = s;
        sp[i + half] = s;
    }
}

static int argmax_mat(const ncnn::Mat& m, int width)
{
    const float* p = (const float*)m.data;
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < width; i++)
    {
        if (p[i] > bestv) { bestv = p[i]; best = i; }
    }
    return best;
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
        if (fv > bestv) { bestv = fv; best = v; }
    }
    return best;
}

static void append_embedding(std::vector<float>& seq, const std::vector<float>& emb, int token, int dim)
{
    const float* row = emb.data() + (size_t)token * dim;
    seq.insert(seq.end(), row, row + dim);
}

static ncnn::Mat codepred_mask(int seq_len)
{
    ncnn::Mat m(seq_len, seq_len, (size_t)4u, 1);
    float* p = m;
    const float neg = -3.4028234663852886e38f;
    for (int y = 0; y < seq_len; y++)
        for (int x = 0; x < seq_len; x++)
            p[y * seq_len + x] = x > y ? neg : 0.0f;
    return m;
}

static ncnn::Mat codepred_pos(int seq_len)
{
    ncnn::Mat m(seq_len, (size_t)4u, 1);
    float* p = m;
    for (int i = 0; i < seq_len; i++) p[i] = (float)i;
    return m;
}

static std::vector<int> run_code_predictor(const std::string& body_dir,
                                           const std::vector<std::vector<float>>& heads,
                                           const std::vector<std::vector<float>>& embs,
                                           const std::vector<float>& talker_codec_embedding,
                                           const float* talker_hidden,
                                           int first_code)
{
    const int dim = 1024;
    const int vocab = 2048;
    std::vector<float> seq(talker_hidden, talker_hidden + dim);
    append_embedding(seq, talker_codec_embedding, first_code, dim);
    std::vector<int> out;
    out.reserve(15);
    for (int step = 0; step < 15; step++)
    {
        const int seq_len = (int)seq.size() / dim;
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), "s%02d", seq_len);
        const std::string param = body_dir + "/code_predictor_body_" + suffix + ".ncnn.param";
        const std::string bin = body_dir + "/code_predictor_body_shared.ncnn.bin";
        ncnn::Net net;
        net.opt.num_threads = 4;
        net.opt.use_vulkan_compute = false;
        net.opt.use_packing_layout = false;
        if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0)
        {
            std::fprintf(stderr, "failed to load code predictor %s\n", suffix);
            std::exit(1);
        }
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", mat2_from_vec(seq, dim, seq_len));
        ex.input("in1", codepred_mask(seq_len));
        ex.input("in2", codepred_pos(seq_len));
        ncnn::Mat hidden;
        if (ex.extract("out0", hidden) != 0)
        {
            std::fprintf(stderr, "code predictor extract failed\n");
            std::exit(1);
        }
        const float* hp = (const float*)hidden.data + (size_t)(seq_len - 1) * dim;
        const int tok = argmax_head(hp, heads[step], vocab, dim);
        out.push_back(tok);
        if (step < 14) append_embedding(seq, embs[step], tok, dim);
    }
    return out;
}

int main(int argc, char** argv)
{
    if (argc != 15)
    {
        std::fprintf(stderr, "Usage: %s <talker_prefill_param> <talker_decode_param> <talker_bin> <prefill_embed> <prefill_mask> <prefill_cos> <prefill_sin> <codepred_body_dir> <codepred_weights_dir> <ref_codes_i32> <frames> <tts_pad_embed_f32> <out_codes_i32> <threads>\n", argv[0]);
        return 2;
    }
    const std::string prefill_param = argv[1];
    const std::string decode_param = argv[2];
    const std::string talker_bin = argv[3];
    const std::string codepred_body_dir = argv[8];
    const std::string weights_dir = argv[9];
    const int frames = std::atoi(argv[11]);
    const int threads = std::atoi(argv[14]) > 0 ? std::atoi(argv[14]) : 4;
    const int dim = 1024;

    ncnn::Net prefill_net;
    prefill_net.opt.num_threads = threads;
    prefill_net.opt.use_vulkan_compute = false;
    prefill_net.opt.use_packing_layout = false;
    ncnn::Net decode_net;
    decode_net.opt.num_threads = threads;
    decode_net.opt.use_vulkan_compute = false;
    decode_net.opt.use_packing_layout = false;
    if (prefill_net.load_param(prefill_param.c_str()) != 0 || prefill_net.load_model(talker_bin.c_str()) != 0 ||
        decode_net.load_param(decode_param.c_str()) != 0 || decode_net.load_model(talker_bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load talker nets\n");
        return 1;
    }

    std::vector<float> heads_flat;
    std::vector<std::vector<float>> heads(15), embs(15);
    std::vector<float> talker_codec_embedding, tts_pad_embed;
    if (!read_f32(weights_dir + "/talker_codec_embedding.f32", talker_codec_embedding) ||
        talker_codec_embedding.size() != (size_t)3072 * dim ||
        !read_f32(argv[12], tts_pad_embed) || tts_pad_embed.size() != (size_t)dim)
    {
        std::fprintf(stderr, "failed to read embeddings\n");
        return 1;
    }
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32(weights_dir + "/lm_head_" + std::to_string(i) + ".f32", heads[i])) return 1;
    }
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32(weights_dir + "/codec_embedding_" + std::to_string(i) + ".f32", embs[i])) return 1;
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> cache;
    ncnn::Mat logits;
    {
        ncnn::Extractor ex = prefill_net.create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", mat2_from_file(argv[4], 1024, 22));
        ex.input("in1", mat2_from_file(argv[5], 22, 22));
        ex.input("in2", mat2_from_file(argv[6], 128, 22));
        ex.input("in3", mat2_from_file(argv[7], 128, 22));
        for (int i = 0; i < 28; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k, v;
            if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return 1;
            cache.emplace_back(std::move(k), std::move(v));
        }
        if (ex.extract("out1", logits) != 0) return 1;
    }

    std::vector<int> generated;
    generated.reserve((size_t)frames * 16);
    ncnn::Mat hidden_last;
    int main_code = argmax_mat(logits.row_range(21, 1), 3072);
    for (int frame = 0; frame < frames; frame++)
    {
        if (frame == 0)
        {
            // Need hidden from prefill for code predictor.
            ncnn::Extractor ex = prefill_net.create_extractor();
            ex.input("in0", mat2_from_file(argv[4], 1024, 22));
            ex.input("in1", mat2_from_file(argv[5], 22, 22));
            ex.input("in2", mat2_from_file(argv[6], 128, 22));
            ex.input("in3", mat2_from_file(argv[7], 128, 22));
            ncnn::Mat h;
            ex.extract("out0", h);
            hidden_last = h.row_range(21, 1).clone();
        }

        std::vector<int> rest = run_code_predictor(codepred_body_dir, heads, embs, talker_codec_embedding, (const float*)hidden_last.data, main_code);
        generated.push_back(main_code);
        for (int t : rest) generated.push_back(t);
        std::printf("frame=%d main=%d rest=", frame, main_code);
        for (int t : rest) std::printf(" %d", t);
        std::printf("\n");

        if (frame + 1 >= frames) break;

        std::vector<float> next_embed;
        append_embedding(next_embed, talker_codec_embedding, main_code, dim);
        for (int i = 0; i < 15; i++) append_embedding(next_embed, embs[i], rest[i], dim);
        std::vector<float> summed(dim, 0.0f);
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < dim; j++) summed[j] += next_embed[(size_t)i * dim + j];
        }
        for (int j = 0; j < dim; j++) summed[j] += tts_pad_embed[j];

        const int position = 22 + frame;
        ncnn::Mat cos, sin;
        make_rope(position, cos, sin);
        ncnn::Mat decode_hidden, decode_logits;
        {
            ncnn::Extractor ex = decode_net.create_extractor();
            ex.set_light_mode(false);
            ex.input("in0", mat2_from_vec(summed, dim, 1));
            ex.input("in1", make_decode_mask(position));
            ex.input("in2", cos);
            ex.input("in3", sin);
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                ex.input(kname, cache[i].first);
                ex.input(vname, cache[i].second);
            }
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                ncnn::Mat k, v;
                if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return 1;
                cache[i] = std::make_pair(std::move(k), std::move(v));
            }
            if (ex.extract("out0", decode_hidden) != 0 || ex.extract("out1", decode_logits) != 0) return 1;
        }
        hidden_last = decode_hidden.clone();
        main_code = argmax_mat(decode_logits, 3072);
    }

    std::vector<int> ref;
    if (read_i32(argv[10], ref))
    {
        int mismatches = 0;
        const int n = std::min((int)generated.size(), (int)ref.size());
        for (int i = 0; i < n; i++) if (generated[i] != ref[i]) mismatches++;
        std::printf("mismatches=%d/%d generated=%zu ref=%zu\n", mismatches, n, generated.size(), ref.size());
    }
    std::ofstream ofs(argv[13], std::ios::binary);
    ofs.write((const char*)generated.data(), (std::streamsize)(generated.size() * sizeof(int)));
    return 0;
}
