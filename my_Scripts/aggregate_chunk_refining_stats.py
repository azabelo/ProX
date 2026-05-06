#!/usr/bin/env python3
"""Sum chunk_refining_stats_rank*.json from a multi-GPU chunk-refining run and print totals."""
import argparse
import glob
import json
import os

SUM_KEYS = (
    "documents",
    "input_tokens",
    "output_tokens",
    "input_lines",
    "output_lines",
    "documents_dropped_empty",
)


def main():
    p = argparse.ArgumentParser(
        description="Aggregate chunk_refining_stats_rank*.json under save_path."
    )
    p.add_argument(
        "save_path",
        help="Directory containing chunk_refining_stats_rank*.json (same as config save_path).",
    )
    args = p.parse_args()
    base = os.path.expanduser(args.save_path)
    paths = sorted(glob.glob(os.path.join(base, "chunk_refining_stats_rank*.json")))
    if not paths:
        raise SystemExit(f"No stats files under {base!r}")

    totals = {k: 0 for k in SUM_KEYS}
    max_wall = 0.0
    max_after = 0.0
    ranks = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        ranks.append(s.get("rank"))
        for k in SUM_KEYS:
            totals[k] += int(s.get(k, 0) or 0)
        max_wall = max(max_wall, float(s.get("wall_seconds_total", 0)))
        max_after = max(max_after, float(s.get("wall_seconds_after_model_load", 0)))

    # Recompute net from summed in/out (more consistent than summing per-rank nets)
    tok_rm = max(0, totals["input_tokens"] - totals["output_tokens"])
    line_rm = max(0, totals["input_lines"] - totals["output_lines"])

    print("=== aggregated chunk refining (all ranks) ===")
    print(f"ranks merged: {len(paths)}  rank ids: {ranks}")
    print(f"documents processed: {totals['documents']}")
    print(f"total input tokens (doc text): {totals['input_tokens']}")
    print(f"total output tokens (refined text): {totals['output_tokens']}")
    print(f"tokens removed (max(0, in - out)): {tok_rm}")
    print(f"total input lines: {totals['input_lines']}")
    print(f"total output lines: {totals['output_lines']}")
    print(f"lines removed (max(0, in - out)): {line_rm}")
    print(f"documents with empty refined output: {totals['documents_dropped_empty']}")
    print(
        "wall time: per-rank wall_seconds_total max (parallel GPUs): "
        f"{max_wall:.3f} s"
    )
    print(
        "wall time after model load (max across ranks): "
        f"{max_after:.3f} s"
    )


if __name__ == "__main__":
    main()
