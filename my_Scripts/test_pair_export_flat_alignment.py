#!/usr/bin/env python3
"""Offline check: flat generate() outputs slice the same way as apply_chunk_refining_pair_export."""
from __future__ import annotations


def collect_rows(batch, chunk_lens, per_doc_chunks, outputs_flat):
    out_idx = 0
    rows: list[tuple[str, str]] = []
    for _sample, n_chunks, chunks in zip(batch, chunk_lens, per_doc_chunks):
        if n_chunks == 0:
            continue
        doc_outs = outputs_flat[out_idx : out_idx + n_chunks]
        out_idx += n_chunks
        for chunk_str, prog in zip(chunks, doc_outs):
            rows.append((chunk_str, prog))
    assert out_idx == len(outputs_flat), (out_idx, len(outputs_flat))
    return rows


def main() -> None:
    batch = [{"text": "a"}, {"text": ""}, {"text": "c"}]
    chunk_lens = [2, 0, 1]
    per_doc_chunks = [["p0", "p1"], [], ["q0"]]
    outputs_flat = ["o0", "o1", "o2"]
    rows = collect_rows(batch, chunk_lens, per_doc_chunks, outputs_flat)
    assert rows == [("p0", "o0"), ("p1", "o1"), ("q0", "o2")]
    print("pair_export_flat_alignment_ok")


if __name__ == "__main__":
    main()
