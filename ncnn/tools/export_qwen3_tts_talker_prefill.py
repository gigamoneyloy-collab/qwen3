#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class TalkerPrefill(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask):
        seq_len = inputs_embeds.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        out = self.model(
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
        logits = self.codec_head(out)
        return out, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text", default="你好，欢迎使用通义千问语音合成。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = tts.model
    talker = model.talker

    input_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])
    input_id = input_ids[0]
    language_id = model.config.talker_config.codec_language_id[args.language.lower()]
    spk_id = model.config.talker_config.spk_id[args.speaker.lower()]
    speaker_embed = talker.get_input_embeddings()(torch.tensor(spk_id, device=input_id.device, dtype=input_id.dtype))

    tts_bos_embed, _tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    codec_prefill = [[
        model.config.talker_config.codec_think_id,
        model.config.talker_config.codec_think_bos_id,
        language_id,
        model.config.talker_config.codec_think_eos_id,
    ]]
    codec0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device=talker.device, dtype=input_id.dtype))
    codec1 = talker.get_input_embeddings()(torch.tensor(
        [[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]],
        device=talker.device,
        dtype=input_id.dtype,
    ))
    codec_input = torch.cat([codec0, speaker_embed.view(1, 1, -1), codec1], dim=1)

    role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    body = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input[:, :-1]
    embeds = torch.cat((role, body), dim=1)
    text_part = torch.cat(
        (
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])),
            _tts_eos_embed,
        ),
        dim=1,
    ) + talker.get_input_embeddings()(
        torch.tensor(
            [[model.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )
    codec_bos = tts_pad_embed + talker.get_input_embeddings()(
        torch.tensor([[model.config.talker_config.codec_bos_id]], device=talker.device, dtype=input_id.dtype)
    )
    embeds = torch.cat((embeds, text_part, codec_bos), dim=1).detach().float()
    mask = torch.ones((1, embeds.shape[1]), dtype=torch.long)

    wrapper = TalkerPrefill(talker).eval()
    with torch.inference_mode():
        hidden, logits = wrapper(embeds, mask)

    embeds.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_prefill_input_f32.bin")
    hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_prefill_hidden_ref_f32.bin")
    logits.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_prefill_logits_ref_f32.bin")

    traced = torch.jit.trace(wrapper, (embeds.clone(), mask), strict=False)
    traced.save(str(out_dir / "talker_prefill.pt"))
    meta = {
        "input_shape": list(embeds.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "argmax_last": int(torch.argmax(logits[:, -1, :], dim=-1).item()),
    }
    (out_dir / "talker_prefill_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
