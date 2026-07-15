#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text", default="你好，欢迎使用通义千问语音合成。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )

    texts = [tts._build_assistant_text(args.text)]
    input_ids = tts._tokenize_texts(texts)
    gen_kwargs = tts._merge_generate_kwargs(
        do_sample=args.sample,
        subtalker_dosample=args.sample,
        max_new_tokens=args.max_new_tokens,
        top_k=1 if not args.sample else 50,
        top_p=1.0,
        temperature=1.0,
    )

    with torch.inference_mode():
        codes_list, hidden_list = tts.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])

    codes = codes_list[0].detach().cpu().numpy().astype(np.int64)
    hidden = hidden_list[0].detach().float().cpu().numpy()
    wav = np.asarray(wavs[0], dtype=np.float32)

    np.save(out_dir / "talker_codes.npy", codes)
    np.save(out_dir / "talker_hidden.npy", hidden)
    sf.write(out_dir / "pytorch_output.wav", wav, sr)

    summary = {
        "model": args.model,
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "sample_rate": int(sr),
        "codes_shape": list(codes.shape),
        "hidden_shape": list(hidden.shape),
        "wav_shape": list(wav.shape),
        "codes_sha256": sha256_array(codes),
        "hidden_sha256": sha256_array(hidden),
        "wav_sha256": sha256_array(wav),
        "gen_kwargs": gen_kwargs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
