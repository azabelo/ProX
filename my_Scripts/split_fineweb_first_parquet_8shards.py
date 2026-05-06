#!/usr/bin/env python3
"""Split first FineWeb sample shard into 8 row shards for multi-GPU doc refining."""
import os

import pyarrow.parquet as pq

SRC = "/home/ubuntu/ProX/data/raw/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
OUT_DIR = "/home/ubuntu/ProX/data/raw/HuggingFaceFW/fineweb/sample/10BT_doc_shards8"
NSHARD = 8


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Reading", SRC, flush=True)
    t = pq.read_table(SRC)
    n = t.num_rows
    for i in range(NSHARD):
        lo = (i * n) // NSHARD
        hi = ((i + 1) * n) // NSHARD
        out = os.path.join(OUT_DIR, f"shard_{i:02d}.parquet")
        pq.write_table(t.slice(lo, hi - lo), out, compression="snappy")
        print(out, "rows", hi - lo, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
