"""
Stage 2 helper: print a quick summary of the *finetuning output* directory.

This is intentionally lightweight: it lists the newest checkpoint and, if present,
the HuggingFace export directory (`hf_ckpt/`) that vLLM expects for stage 3.

Example:

  python finetune_refiner/print_output.py --out-dir finetuning_refiner/outputs/qwen05b_chunk_program_pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_latest_checkpoint_dir(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.is_dir():
        return None
    candidates = [p for p in checkpoints_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Prefer global_step_* numeric, else newest mtime.
    def key(p: Path):
        name = p.name
        if name.startswith("global_step_"):
            try:
                return (0, -int(name.split("_")[-1]))
            except Exception:
                return (1, 0)
        return (2, -int(p.stat().st_mtime))

    return sorted(candidates, key=key)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"Not a directory: {out_dir}")

    ckpts = out_dir / "checkpoints"
    latest = _find_latest_checkpoint_dir(ckpts)
    print(f"out_dir: {out_dir}")
    if latest is None:
        print("checkpoints: (none found)")
        return

    print(f"latest_checkpoint: {latest}")
    hf = latest / "hf_ckpt"
    if hf.is_dir():
        cfg = hf / "config.json"
        print(f"hf_ckpt: {hf} ({'has config.json' if cfg.is_file() else 'missing config.json'})")
    else:
        print("hf_ckpt: (missing)  # stage-3 vLLM needs an HF export under .../hf_ckpt/")


if __name__ == "__main__":
    main()

