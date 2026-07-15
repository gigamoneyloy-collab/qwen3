#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class CodePredictorPrefill(torch.nn.Module):
    def __init__(self, talker, generation_steps: int = 14):
        super().__init__()
        self.predictor = talker.code_predictor
        self.generation_steps = generation_steps

    def forward(self, inputs_embeds):
        seq_len = inputs_embeds.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, seq_len)
        out = self.predictor.model(
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
        logits = self.predictor.lm_head[self.generation_steps](out)
        return out, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--talker-hidden", required=True)
    parser.add_argument("--first-code", type=int, default=-1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker

    h = np.fromfile(args.talker_hidden, dtype=np.float32).reshape(1, 10, 1024)
    past_hidden = torch.from_numpy(h[:, -1:, :]).to(talker.device)
    if args.first_code >= 0:
        first_code = args.first_code
    else:
        logits = np.fromfile(out_dir / "talker_prefill_logits_ref_f32.bin", dtype=np.float32).reshape(1, 10, 3072)
        first_code = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    first_id = torch.tensor([[first_code]], dtype=torch.long, device=talker.device)
    last_id_hidden = talker.get_input_embeddings()(first_id)
    inputs = torch.cat((past_hidden, last_id_hidden), dim=1).detach().float()

    wrapper = CodePredictorPrefill(talker, generation_steps=14).eval()
    with torch.inference_mode():
        hidden, logits = wrapper(inputs)

    inputs.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_prefill_input_f32.bin")
    hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_prefill_hidden_ref_f32.bin")
    logits.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_prefill_logits_ref_f32.bin")
    traced = torch.jit.trace(wrapper, inputs.clone(), strict=False)
    traced.save(str(out_dir / "code_predictor_prefill.pt"))
    meta = {
        "input_shape": list(inputs.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "first_code": first_code,
        "argmax_last": int(torch.argmax(logits[:, -1, :], dim=-1).item()),
    }
    (out_dir / "code_predictor_prefill_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
