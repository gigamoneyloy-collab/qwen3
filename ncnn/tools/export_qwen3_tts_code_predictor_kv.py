#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class CodePredictorStep(torch.nn.Module):
    def __init__(self, talker, step: int):
        super().__init__()
        self.predictor = talker.code_predictor
        self.step = int(step)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        seq_len = inputs_embeds.shape[1]
        hidden = self.predictor.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=torch.arange(seq_len, device=inputs_embeds.device),
        ).last_hidden_state
        last_hidden = hidden[:, -1:, :]
        logits = self.predictor.lm_head[self.step](last_hidden)
        return last_hidden, logits


def causal_mask(seq_len: int, device):
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32, device=device)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


def zero_decode_mask(kv_len: int, device):
    return torch.zeros((1, 1, 1, kv_len + 1), dtype=torch.float32, device=device)


def trace_one(module, inputs, path: Path):
    traced = torch.jit.trace(module, inputs, strict=False)
    traced.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--talker-hidden", required=True)
    ap.add_argument("--first-code", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode-steps", default="1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    predictor = talker.code_predictor
    device = talker.device

    h_flat = np.fromfile(args.talker_hidden, dtype=np.float32)
    h = h_flat.reshape(1, h_flat.size // 1024, 1024)
    past_hidden = torch.from_numpy(h[:, -1:, :]).to(device)
    first_code = torch.tensor([[args.first_code]], dtype=torch.long, device=device)
    first_emb = talker.get_input_embeddings()(first_code)

    prefill_inputs = torch.cat((past_hidden, first_emb), dim=1).detach().float()
    prefill_mask = causal_mask(2, device)
    prefill_pos = torch.arange(2, dtype=torch.long, device=device).view(1, 2)
    prefill = CodePredictorStep(talker, 0).eval()

    with torch.inference_mode():
        prefill_last_hidden, prefill_logits = prefill(prefill_inputs, prefill_mask, prefill_pos)
        token0 = torch.argmax(prefill_logits, dim=-1)

    trace_one(prefill, (prefill_inputs.clone(), prefill_mask.clone(), prefill_pos.clone()), out_dir / "codepred_prefill_s2_step00.pt")
    prefill_inputs.cpu().numpy().astype(np.float32).tofile(out_dir / "codepred_prefill_s2_input_f32.bin")
    prefill_mask.cpu().numpy().astype(np.float32).tofile(out_dir / "codepred_prefill_s2_mask_f32.bin")
    prefill_pos.cpu().numpy().astype(np.int64).tofile(out_dir / "codepred_prefill_s2_pos_i64.bin")
    prefill_last_hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "codepred_prefill_s2_last_hidden_ref_f32.bin")
    prefill_logits.cpu().numpy().astype(np.float32).tofile(out_dir / "codepred_prefill_s2_logits_ref_f32.bin")

    # Produce PyTorch decode references for requested steps using the same no-cache
    # prefix math as a reference, while tracing single-token decode graphs that will
    # receive KV cache after ncnn SDPA patching.
    seq = prefill_inputs.clone()
    tokens = [int(token0.item())]
    decode_steps = [int(x) for x in args.decode_steps.split(",") if x.strip()]
    decode_meta = []
    for step in range(1, 15):
        emb = predictor.get_input_embeddings()[step - 1](torch.tensor([[tokens[-1]]], dtype=torch.long, device=device))
        seq = torch.cat((seq, emb), dim=1)
        full = CodePredictorStep(talker, step).eval()
        full_mask = causal_mask(seq.shape[1], device)
        full_pos = torch.arange(seq.shape[1], dtype=torch.long, device=device).view(1, seq.shape[1])
        with torch.inference_mode():
            full_last_hidden, full_logits = full(seq, full_mask, full_pos)
            tok = torch.argmax(full_logits, dim=-1)
        tokens.append(int(tok.item()))

        if step in decode_steps:
            dec = CodePredictorStep(talker, step).eval()
            dec_input = emb.detach().float()
            dec_mask = zero_decode_mask(step + 1, device)
            dec_pos = torch.tensor([[step + 1]], dtype=torch.long, device=device)
            trace_one(dec, (dec_input.clone(), dec_mask.clone(), dec_pos.clone()), out_dir / f"codepred_decode_s1_step{step:02d}.pt")
            dec_input.cpu().numpy().astype(np.float32).tofile(out_dir / f"codepred_decode_s1_step{step:02d}_input_f32.bin")
            dec_mask.cpu().numpy().astype(np.float32).tofile(out_dir / f"codepred_decode_s1_step{step:02d}_mask_f32.bin")
            dec_pos.cpu().numpy().astype(np.int64).tofile(out_dir / f"codepred_decode_s1_step{step:02d}_pos_i64.bin")
            full_last_hidden.cpu().numpy().astype(np.float32).tofile(out_dir / f"codepred_decode_s1_step{step:02d}_last_hidden_ref_f32.bin")
            full_logits.cpu().numpy().astype(np.float32).tofile(out_dir / f"codepred_decode_s1_step{step:02d}_logits_ref_f32.bin")
            decode_meta.append({"step": step, "token": int(tok.item()), "seq_len": int(seq.shape[1])})

    meta = {
        "prefill_input_shape": list(prefill_inputs.shape),
        "prefill_mask_shape": list(prefill_mask.shape),
        "prefill_pos_shape": list(prefill_pos.shape),
        "prefill_token0": int(token0.item()),
        "tokens": tokens,
        "decode": decode_meta,
    }
    (out_dir / "codepred_kv_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
