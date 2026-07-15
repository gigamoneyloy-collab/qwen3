#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param")
    ap.add_argument("--backup-suffix", default=".tensor_to")
    args = ap.parse_args()
    path = Path(args.param)
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise SystemExit(f"not an ncnn param: {path}")
    backup = path.with_name(path.name + args.backup_suffix)
    if not backup.exists():
        backup.write_text(raw, encoding="utf-8")

    body = lines[2:]
    replace = {}
    kept = []
    removed = 0
    for line in body:
        f = line.split()
        if f and f[0] == "Tensor.to":
            nin = int(f[2])
            nout = int(f[3])
            if nin != 1 or nout != 1:
                raise SystemExit(f"unexpected Tensor.to shape: {line}")
            inp = f[4]
            out = f[5]
            replace[out] = inp
            removed += 1
            continue
        kept.append(line)

    rewritten = []
    for line in kept:
        f = line.split()
        if not f:
            rewritten.append(line)
            continue
        if len(f) >= 4:
            nin = int(f[2])
            nout = int(f[3])
            blob_end = min(len(f), 4 + nin + nout)
            for i in range(4, blob_end):
                if f[i] in replace:
                    f[i] = replace[f[i]]
        rewritten.append(" ".join(f))

    def noutputs(line):
        f = line.split()
        return int(f[3]) if len(f) >= 4 else 0

    new_lines = [lines[0], f"{len(rewritten)} {sum(noutputs(x) for x in rewritten)}"] + rewritten
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"patched {path}: removed Tensor.to={removed}")


if __name__ == "__main__":
    main()
