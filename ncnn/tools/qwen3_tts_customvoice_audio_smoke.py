#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct
import subprocess
import time
import wave
from pathlib import Path

from qwen3_tts_customvoice_prompt_matrix import DEFAULT_CASES, maybe_windows_path


def wav_i16_stats(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if channels != 1 or sample_width != 2:
        raise ValueError(f"expected mono s16 wav, got channels={channels} width={sample_width}")
    count = len(raw) // 2
    if count == 0:
        return {
            "samples": 0,
            "sample_rate": sample_rate,
            "max_abs": 0,
            "rms": 0.0,
        }
    vals = struct.unpack("<" + "h" * count, raw)
    max_abs = max(abs(v) for v in vals)
    rms = (sum(float(v) * float(v) for v in vals) / count) ** 0.5
    return {
        "samples": count,
        "sample_rate": sample_rate,
        "max_abs": int(max_abs),
        "rms": float(rms),
    }


def parse_prompt_len(stdout):
    m = re.search(r"frontend prompt_len=(\d+)", stdout)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3-tts-exe", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--vulkan", action="store_true")
    ap.add_argument("--min-audible-rms", type=float, default=100.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8")) if args.cases else DEFAULT_CASES
    use_windows_paths = os.name != "nt" and args.qwen3_tts_exe.lower().endswith(".exe")

    rows = []
    for case in cases:
        case_id = case["id"]
        text_path = out_dir / f"{case_id}_text.txt"
        wav_path = out_dir / f"{case_id}_{args.frames}f.wav"
        codes_path = out_dir / f"{case_id}_{args.frames}f_codes_i32.bin"
        text_path.write_text(case["text"], encoding="utf-8")

        cmd = [
            args.qwen3_tts_exe,
            "--model",
            maybe_windows_path(Path(args.model_json), use_windows_paths),
            "--frames",
            str(args.frames),
            "--out",
            maybe_windows_path(wav_path, use_windows_paths),
            "--codes",
            maybe_windows_path(codes_path, use_windows_paths),
            "--threads",
            str(args.threads),
            "--text-file",
            maybe_windows_path(text_path, use_windows_paths),
            "--speaker",
            case["speaker"],
            "--language",
            case["language"],
        ]
        if args.vulkan:
            cmd.append("--vulkan")

        t0 = time.perf_counter()
        cp = subprocess.run(cmd, text=True, capture_output=True)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        stats = None
        wav_ok = False
        audible = False
        error = ""
        if cp.returncode == 0 and wav_path.exists():
            try:
                stats = wav_i16_stats(wav_path)
                wav_ok = stats["samples"] == args.frames * 1920
                audible = stats["rms"] >= args.min_audible_rms
            except Exception as exc:
                error = str(exc)
        elif cp.stderr:
            error = cp.stderr.strip()

        row = {
            **case,
            "frames": args.frames,
            "prompt_len": parse_prompt_len(cp.stdout),
            "returncode": cp.returncode,
            "wall_ms": wall_ms,
            "wav_ok": wav_ok,
            "audible": audible,
            "wav": str(wav_path),
            "codes": str(codes_path),
            "stats": stats,
        }
        if error:
            row["error"] = error
        if cp.stdout:
            row["stdout"] = cp.stdout.strip()
        if cp.stderr:
            row["stderr"] = cp.stderr.strip()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "total": len(rows),
        "generated": sum(1 for r in rows if r["returncode"] == 0),
        "wav_ok": sum(1 for r in rows if r["wav_ok"]),
        "audible": sum(1 for r in rows if r["audible"]),
        "frames": args.frames,
        "min_audible_rms": args.min_audible_rms,
        "cases": rows,
    }
    (out_dir / "customvoice_audio_smoke.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if summary["generated"] != summary["total"] or summary["wav_ok"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
