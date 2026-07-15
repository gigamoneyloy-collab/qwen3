#include "qwen_bpe.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

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
    if (argc != 3)
    {
        std::fprintf(stderr, "Usage: %s <tokenizer.txt> <text>\n", argv[0]);
        return 2;
    }
    try
    {
        q3tts::QwenBpe bpe;
        bpe.load(argv[1]);
        std::string text;
        if (!read_text_arg(argv[2], text))
        {
            std::fprintf(stderr, "failed to read text %s\n", argv[2]);
            return 1;
        }
        std::vector<int> ids = bpe.encode(text);
        std::printf("count=%zu\n", ids.size());
        for (size_t i = 0; i < ids.size(); i++)
        {
            if (i) std::printf(" ");
            std::printf("%d", ids[i]);
        }
        std::printf("\n");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
