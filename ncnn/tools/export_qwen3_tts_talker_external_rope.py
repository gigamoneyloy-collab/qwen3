#!/usr/bin/env python3
import argparse
import json
import json
from pathlib import Path

import torch

from qwen_tts import Qwen3TTSModel
from export_qwen3_tts_talker_next import build_nonstream_prefill


class TalkerExternalRope(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, cos_cache, sin_cache):
        hidden_states = inputs_embeds
        position_embeddings = (cos_cache, sin_cache)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]
        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states)
        return hidden_states, logits


def causal_mask(seq_len: int, device):
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32, device=device)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--text", default="你好，欢迎使用通义千问语音合成。")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    embeds, _ = build_nonstream_prefill(tts, args.text, args.language, args.speaker)
    embeds = embeds.detach().float()
    seq_len = embeds.shape[1]
    mask = causal_mask(seq_len, embeds.device)
    position_ids = torch.arange(seq_len, device=embeds.device).view(1, seq_len)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        cos, sin = talker.model.rotary_emb(embeds, position_ids)

    wrapper = TalkerExternalRope(talker).eval()
    with torch.inference_mode():
        hidden, logits = wrapper(embeds, mask, cos, sin)

    embeds.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_input_f32.bin")
    mask.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_mask_f32.bin")
    cos.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_cos_f32.bin")
    sin.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_sin_f32.bin")
    hidden.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_hidden_ref_f32.bin")
    logits.cpu().numpy().astype("float32").tofile(out_dir / "talker_external_logits_ref_f32.bin")

    traced = torch.jit.trace(wrapper, (embeds.clone(), mask.clone(), cos.clone(), sin.clone()), strict=False)
    traced.save(str(out_dir / "talker_external_rope_s22.pt"))
    meta = {
        "input_shape": list(embeds.shape),
        "mask_shape": list(mask.shape),
        "cos_shape": list(cos.shape),
        "sin_shape": list(sin.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "argmax_last": int(torch.argmax(logits[:, -1, :], dim=-1).item()),
    }
    (out_dir / "talker_external_rope_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
