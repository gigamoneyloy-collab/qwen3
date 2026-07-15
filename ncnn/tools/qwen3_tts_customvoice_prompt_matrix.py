#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
from export_qwen3_tts_talker_next import build_nonstream_prefill


DEFAULT_CASES = [
    {
        "id": "en_vivian",
        "text": "Today we are testing clear English speech.",
        "language": "English",
        "speaker": "Vivian",
    },
    {
        "id": "zh_vivian",
        "text": "今天我们测试一段清晰的中文语音。",
        "language": "Chinese",
        "speaker": "Vivian",
    },
    {
        "id": "ja_ono_anna",
        "text": "今日は日本語の音声合成をテストしています。",
        "language": "Japanese",
        "speaker": "Ono_Anna",
    },
    {
        "id": "ko_sohee",
        "text": "오늘은 한국어 음성 합성을 테스트하고 있습니다.",
        "language": "Korean",
        "speaker": "Sohee",
    },
    {
        "id": "de_eric",
        "text": "Heute testen wir eine klare deutsche Sprachsynthese.",
        "language": "German",
        "speaker": "Eric",
    },
    {
        "id": "fr_serena",
        "text": "Nous testons aujourd'hui une synthese vocale francaise claire.",
        "language": "French",
        "speaker": "Serena",
    },
    {
        "id": "ru_aiden",
        "text": "Segodnya my proveriaem russkuyu sintezirovannuyu rech.",
        "language": "Russian",
        "speaker": "Aiden",
    },
    {
        "id": "pt_ryan",
        "text": "Hoje estamos testando uma fala portuguesa clara.",
        "language": "Portuguese",
        "speaker": "Ryan",
    },
    {
        "id": "es_dylan",
        "text": "Hoy probamos una sintesis de voz espanola clara.",
        "language": "Spanish",
        "speaker": "Dylan",
    },
    {
        "id": "it_uncle_fu",
        "text": "Oggi testiamo una sintesi vocale italiana chiara.",
        "language": "Italian",
        "speaker": "Uncle_Fu",
    },
    {
        "id": "zh_eric_dialect",
        "text": "今天我们测试中文方言说话人的提示构造。",
        "language": "Chinese",
        "speaker": "Eric",
    },
    {
        "id": "zh_dylan_dialect",
        "text": "今天我们继续测试中文方言说话人的提示构造。",
        "language": "Chinese",
        "speaker": "Dylan",
    },
]


def parse_frontend_check(stdout):
    shape = None
    metrics = {}
    m = re.search(r"prompt_shape=\((\d+),(\d+)\)", stdout)
    if m:
        shape = [int(m.group(1)), int(m.group(2))]
    m = re.search(r"prompt mae=([0-9.eE+-]+) rmse=([0-9.eE+-]+) maxe=([0-9.eE+-]+)", stdout)
    if m:
        metrics = {
            "mae": float(m.group(1)),
            "rmse": float(m.group(2)),
            "maxe": float(m.group(3)),
        }
    return shape, metrics


def maybe_windows_path(path, use_windows_paths):
    s = str(path)
    if not use_windows_paths:
        return s
    cp = subprocess.run(["wslpath", "-w", s], text=True, capture_output=True, check=True)
    return cp.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frontend-dir", required=True)
    ap.add_argument("--frontend-check-exe", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-error", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontend_dir = Path(args.frontend_dir)
    use_windows_paths = os.name != "nt" and args.frontend_check_exe.lower().endswith(".exe")

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8")) if args.cases else DEFAULT_CASES

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    rows = []
    for case in cases:
        case_id = case["id"]
        ref_path = out_dir / f"{case_id}_prompt_ref_f32.bin"
        text_path = out_dir / f"{case_id}_text.txt"
        text_path.write_text(case["text"], encoding="utf-8")
        prefill, _ = build_nonstream_prefill(tts, case["text"], case["language"], case["speaker"])
        arr = prefill.detach().cpu().numpy().astype(np.float32)
        arr.tofile(ref_path)

        cmd = [
            args.frontend_check_exe,
            maybe_windows_path(frontend_dir / "tokenizer.txt", use_windows_paths),
            maybe_windows_path(frontend_dir / "talker_text_embed.ncnn.param", use_windows_paths),
            maybe_windows_path(frontend_dir / "talker_text_embed.ncnn.bin", use_windows_paths),
            maybe_windows_path(frontend_dir / "talker_codec_embed.ncnn.param", use_windows_paths),
            maybe_windows_path(frontend_dir / "talker_codec_embed.ncnn.bin", use_windows_paths),
            "@" + maybe_windows_path(text_path, use_windows_paths),
            case["language"],
            case["speaker"],
            maybe_windows_path(ref_path, use_windows_paths),
        ]
        cp = subprocess.run(cmd, text=True, capture_output=True)
        shape, metrics = parse_frontend_check(cp.stdout)
        passed = cp.returncode == 0 and metrics.get("maxe", float("inf")) <= args.max_error
        row = {
            **case,
            "ref_shape": list(arr.shape),
            "cpp_shape": shape,
            "metrics": metrics,
            "passed": passed,
            "returncode": cp.returncode,
        }
        if cp.stderr:
            row["stderr"] = cp.stderr.strip()
        if cp.stdout:
            row["stdout"] = cp.stdout.strip()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r["passed"]),
        "failed": sum(1 for r in rows if not r["passed"]),
        "max_error_threshold": args.max_error,
        "cases": rows,
    }
    (out_dir / "customvoice_prompt_matrix.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
