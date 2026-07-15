#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
from export_qwen3_tts_talker_next import TalkerFull, build_nonstream_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--text", default="你好，欢迎使用通义千问语音合成。")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "talker_body_by_len"
    traces_dir.mkdir(exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    prefill, tts_pad_embed = build_nonstream_prefill(tts, args.text, args.language, args.speaker)
    body = TalkerFull(talker).eval()

    seq = prefill.detach().float()
    all_codes = []
    all_main = []
    for frame in range(args.steps):
        seq_len = seq.shape[1]
        seq.cpu().numpy().astype(np.float32).tofile(traces_dir / f"input_s{seq_len:03d}_f32.bin")
        traced = torch.jit.trace(body, seq.clone(), strict=False)
        traced.save(str(traces_dir / f"talker_body_s{seq_len:03d}.pt"))
        with torch.no_grad():
            hidden, logits = body(seq)
            main = torch.argmax(logits[:, -1, :], dim=-1)
            pred = talker.code_predictor.generate(
                inputs_embeds=torch.cat((hidden[:, -1:, :], talker.get_input_embeddings()(main[:, None])), dim=1),
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=False,
                top_p=0.00001,
                top_k=1,
                temperature=0.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            frame_codes = torch.cat((main[:, None], pred.sequences), dim=-1)
        all_main.append(int(main.item()))
        all_codes.append(frame_codes.cpu().numpy().astype(np.int32).reshape(-1).tolist())
        hidden.cpu().numpy().astype(np.float32).tofile(traces_dir / f"hidden_s{seq_len:03d}_ref_f32.bin")
        logits.cpu().numpy().astype(np.float32).tofile(traces_dir / f"logits_s{seq_len:03d}_ref_f32.bin")
        if frame + 1 < args.steps:
            pieces = [talker.get_input_embeddings()(frame_codes[:, :1])]
            for i in range(15):
                pieces.append(talker.code_predictor.get_input_embeddings()[i](frame_codes[:, i + 1:i + 2]))
            next_embed = torch.cat(pieces, dim=1).sum(1, keepdim=True) + tts_pad_embed
            seq = torch.cat((seq, next_embed), dim=1).detach().float()

    np.asarray(all_codes, dtype=np.int32).tofile(out_dir / "talker_multistep_codes_ref_i32.bin")
    meta = {
        "steps": args.steps,
        "main_codes": all_main,
        "codes": all_codes,
        "seq_lens": [22 + i for i in range(args.steps)],
    }
    (out_dir / "talker_multistep_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
