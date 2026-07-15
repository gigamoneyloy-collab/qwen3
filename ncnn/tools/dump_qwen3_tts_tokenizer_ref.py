#!/usr/bin/env python3
import argparse
from pathlib import Path

from qwen_tts import Qwen3TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.text
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype="float32",
        attn_implementation="eager",
    )
    ids = model._tokenize_texts([text])[0].reshape(-1).tolist()
    print(f"count={len(ids)}")
    print(" ".join(str(int(x)) for x in ids))


if __name__ == "__main__":
    main()
