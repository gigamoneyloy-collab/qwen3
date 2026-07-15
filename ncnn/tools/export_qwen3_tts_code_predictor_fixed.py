#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class CodePredictorFixed(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.predictor = talker.code_predictor
        self.emb0 = talker.get_input_embeddings()
        self.embs = talker.code_predictor.get_input_embeddings()

    def _forward_step(self, inputs_embeds, step: int):
        seq_len = inputs_embeds.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, seq_len)
        hidden = self.predictor.model(
            input_ids=None,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=torch.arange(seq_len, device=inputs_embeds.device),
        ).last_hidden_state
        logits = self.predictor.lm_head[step](hidden[:, -1:, :])
        token = torch.argmax(logits, dim=-1)
        return hidden[:, -1:, :], logits, token

    def forward(self, talker_hidden, first_code):
        last_id_hidden = self.emb0(first_code)
        seq = torch.cat((talker_hidden, last_id_hidden), dim=1)
        logits_out = []
        tokens = []
        for step in range(15):
            _, logits, token = self._forward_step(seq, step)
            logits_out.append(logits)
            tokens.append(token)
            if step < 14:
                seq = torch.cat((seq, self.embs[step](token)), dim=1)
        return torch.cat(logits_out, dim=1), torch.cat(tokens, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--talker-hidden", required=True)
    ap.add_argument("--first-code", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    wrapper = CodePredictorFixed(tts.model.talker).eval()
    h_flat = np.fromfile(args.talker_hidden, dtype=np.float32)
    h = h_flat.reshape(1, h_flat.size // 1024, 1024)
    talker_hidden = torch.from_numpy(h[:, -1:, :]).to(tts.model.talker.device)
    first_code = torch.tensor([[args.first_code]], dtype=torch.long, device=tts.model.talker.device)
    with torch.inference_mode():
        logits, tokens = wrapper(talker_hidden, first_code)
    talker_hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_fixed_hidden_input_f32.bin")
    first_code.cpu().numpy().astype(np.int32).tofile(out_dir / "code_predictor_fixed_first_code_i32.bin")
    logits.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_fixed_logits_ref_f32.bin")
    tokens.cpu().numpy().astype(np.int32).tofile(out_dir / "code_predictor_fixed_tokens_ref_i32.bin")
    traced = torch.jit.trace(wrapper, (talker_hidden.clone(), first_code.clone()), strict=False)
    traced.save(str(out_dir / "code_predictor_fixed.pt"))
    meta = {
        "hidden_input_shape": list(talker_hidden.shape),
        "first_code_shape": list(first_code.shape),
        "logits_shape": list(logits.shape),
        "tokens_shape": list(tokens.shape),
        "first_code": args.first_code,
        "tokens": tokens.cpu().numpy().astype(int).reshape(-1).tolist(),
    }
    (out_dir / "code_predictor_fixed_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
