#!/usr/bin/env python3
"""
Merge chunk-refining outputs (prox_<rank>/prox_*.parquet) into one Parquet file.

Shards are concatenated in global document order: rank 0 files (sorted by name),
then rank 1, … This matches apply_chunk_refining row-sharding when TOTAL_SPLIT > 1.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def iter_rank_dirs(base: Path) -> list[tuple[int, Path]]:
    dirs: list[tuple[int, Path]] = []
    for p in base.iterdir():
        m = re.fullmatch(r"prox_(\d+)", p.name)
        if p.is_dir() and m:
            dirs.append((int(m.group(1)), p))
    return sorted(dirs, key=lambda x: x[0])


def iter_shard_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for _, d in iter_rank_dirs(base):
        files.extend(sorted(d.glob("prox_*.parquet")))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory that contains prox_0, prox_1, … (same as config save_path).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path for the single merged .parquet file.",
    )
    ap.add_argument(
        "--batch-rows",
        type=int,
        default=8192,
        help="Row groups read in batches of this many rows per read.",
    )
    args = ap.parse_args()

    base = Path(args.input_dir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    shard_files = iter_shard_files(base)
    if not shard_files:
        raise SystemExit(
            f"No prox_*/prox_*.parquet found under {base}\n"
            "Expect one directory per rank (prox_0, prox_1, …) with prox_*.parquet shards. "
            "If workers finished but wrote nothing, check: (1) per-rank logs for crashes before the first save, "
            "(2) LIMIT vs NGPU row sharding (small LIMIT with many GPUs can leave every rank with zero rows), "
            "(3) input rows all empty so no pair rows were emitted."
        )

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for fp in shard_files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=args.batch_rows):
                if batch.num_rows == 0:
                    continue
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        out.as_posix(),
                        table.schema,
                        compression="zstd",
                        version="2.6",
                    )
                writer.write_table(table)
                total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        if out.exists():
            out.unlink()
        raise SystemExit("No rows written; refusing to create an empty parquet.")

    print(f"Wrote {out} ({total_rows} rows) from {len(shard_files)} shard file(s).")


if __name__ == "__main__":
    main()
