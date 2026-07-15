#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="Vivian")
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--out-wav", required=True)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    p.add_argument("--attn", default=None)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    args = parse_args()
    load_kwargs = {}
    if args.device == "cuda":
        load_kwargs["device_map"] = "cuda:0"
        load_kwargs["dtype"] = torch.bfloat16 if args.dtype != "float32" else torch.float32
        if args.attn:
            load_kwargs["attn_implementation"] = args.attn
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["dtype"] = torch.float32

    t0 = time.perf_counter()
    tts = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    sync(args.device)
    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        subtalker_dosample=False,
    )
    sync(args.device)
    gen_s = time.perf_counter() - t0

    wav = np.asarray(wavs[0], dtype=np.float32)
    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, wav, sr)

    audio_s = float(wav.shape[0]) / float(sr)
    result = {
        "backend": f"pytorch-{args.device}",
        "load_s": load_s,
        "generate_s": gen_s,
        "sample_rate": int(sr),
        "samples": int(wav.shape[0]),
        "audio_s": audio_s,
        "rtf": gen_s / audio_s if audio_s > 0 else None,
        "max_new_tokens": args.max_new_tokens,
        "dtype": str(load_kwargs.get("dtype", "float32")),
        "attn": args.attn,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
