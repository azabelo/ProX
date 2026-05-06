"""
Chunk LM inference that writes **one parquet row per chunk**: chunk text (input) × model
output for that chunk (target). Mirrors ``apply_chunk_refining`` IO / sharding;
does not merge chunks or run ``execute_meta_operations``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_utils import get_adapter_func, split_into_batches
from vllm import LLM, SamplingParams

from data_gen.configs import GentaskConfig
from data_gen.utils.hf_tokenizer import load_auto_tokenizer_fast_fallback

# Rank / TOTAL_SPLIT from env (same as apply_chunk_refining).
import data_gen.tasks.apply_chunk_refining as _cr

RANK = _cr.RANK
TOTAL_SPLIT = _cr.TOTAL_SPLIT

trunc_text = _cr.trunc_text
_stream_window_size = _cr._stream_window_size
_split_size = _cr._split_size
_sorted_parquet_paths = _cr._sorted_parquet_paths
_parquet_total_rows = _cr._parquet_total_rows
_iter_parquet_rank_row_shard = _cr._iter_parquet_rank_row_shard
_doc_input_tokens = _cr._doc_input_tokens


def _tqdm_if(enable: bool, it, **kwargs):
    if not enable:
        return it
    return tqdm(it, **kwargs)


def _process_pair_batches(
    arguments,
    *,
    engine,
    tokenizer,
    sampling_params,
    args,
    config,
    dir_path,
    base_name,
    file_seq_start,
    split_sz,
    show_progress: bool = True,
):
    """Per-chunk rows: ``text`` = chunk body (line-numbered), ``target`` = vLLM output."""
    _ = (args, config)  # same signature as ``_process_batches``; no meta-ops on pairs
    # Single list copy when one sub-batch: avoid range() slice list materialization.
    if split_sz and len(arguments) <= split_sz:
        batches = [arguments]
    else:
        batches = split_into_batches(arguments, split_sz)
    file_seq = file_seq_start
    d_docs = 0
    d_pairs = 0
    d_in_tok = 0
    d_out_tok = 0
    d_in_lines = 0
    d_out_lines = 0
    d_empty_targets = 0

    for _bi, batch in enumerate(
        _tqdm_if(show_progress, batches, desc="save_interval batches")
    ):
        prompts: list[str] = []
        chunk_lens: list[int] = []
        per_doc_chunks: list[list[str]] = []

        for sample in _tqdm_if(
            show_progress, batch, total=len(batch), unit="tokenizing", leave=False
        ):
            raw = sample.get("text") or ""
            d_in_tok += _doc_input_tokens(tokenizer, raw)
            d_in_lines += len(raw.splitlines())

            if sample["text"] == "":
                chunk_lens.append(0)
                per_doc_chunks.append([])
                continue

            user_msgs = trunc_text(sample["text"], tokenizer, max_token=1500)
            chunk_lens.append(len(user_msgs))
            per_doc_chunks.append(user_msgs)
            for user_msg in user_msgs:
                user_msg = f"[doc]\n{user_msg}\n[/doc]"
                total_msg = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant.",
                        },
                        {"role": "user", "content": user_msg},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(total_msg)

        if not prompts:
            outputs_flat: list[str] = []
        else:
            gen = engine.generate(prompts, sampling_params)
            outputs_flat = [item.outputs[0].text.strip(" ") for item in gen]

        out_idx = 0
        pair_rows: list[dict] = []
        for sample, n_chunks, chunks in zip(batch, chunk_lens, per_doc_chunks):
            if n_chunks == 0:
                continue
            doc_outs = outputs_flat[out_idx : out_idx + n_chunks]
            out_idx += n_chunks
            d_docs += 1
            meta = sample.get("metadata") or {}
            doc_program = meta.get("doc_program", "")
            for local_i, (chunk_str, prog) in enumerate(zip(chunks, doc_outs)):
                d_pairs += 1
                if not (prog or "").strip():
                    d_empty_targets += 1
                ot = _doc_input_tokens(tokenizer, prog) if prog else 0
                ol = len(prog.splitlines()) if prog else 0
                d_out_tok += ot
                d_out_lines += ol
                pair_rows.append(
                    {
                        "text": chunk_str,
                        "target": prog,
                        "chunk_index": local_i,
                        "doc_program": doc_program,
                    }
                )

        assert out_idx == len(outputs_flat), (
            f"output alignment bug: used {out_idx} vs {len(outputs_flat)} outputs"
        )

        if not pair_rows:
            continue

        intermediate_ds = Dataset.from_list(pair_rows)
        file_seq += 1
        out_path = os.path.join(dir_path, f"{base_name}_{file_seq:06d}.parquet")
        intermediate_ds.to_parquet(out_path)

    return file_seq, {
        "documents": d_docs,
        "pairs": d_pairs,
        "input_tokens": d_in_tok,
        "output_tokens": d_out_tok,
        "input_lines": d_in_lines,
        "output_lines": d_out_lines,
        "pairs_with_empty_target": d_empty_targets,
    }


def main(args):
    t_wall0 = time.perf_counter()
    config = GentaskConfig().from_yaml(args.config_path)
    if getattr(args, "save_path", None) and str(args.save_path).strip():
        config.save_path = str(args.save_path).strip()
    print(
        f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] startup: loading tokenizer ({args.model_path}) …",
        flush=True,
    )
    tokenizer, tokenizer_slow_fallback = load_auto_tokenizer_fast_fallback(args.model_path)
    if tokenizer_slow_fallback:
        print(
            f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] fast tokenizer unavailable; "
            f"using slow tokenizer and vLLM tokenizer_mode=slow.",
            flush=True,
        )
    print(
        f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] tokenizer ready; data_path={config.data_path!r}",
        flush=True,
    )

    data_reader = None
    doc_iter = None
    data_folder = None
    paths = None
    parquet_row_shard = False
    total_rows = 0
    total_effective = 0
    row_start = 0
    row_end = 0
    if args.data_format == "parquet":
        data_folder = get_datafolder(config.data_path)
        paths = _sorted_parquet_paths(data_folder)
        if not paths:
            raise RuntimeError(f"No parquet files under {config.data_path!r}")

        if TOTAL_SPLIT > 1:
            parquet_row_shard = True
            total_rows = _parquet_total_rows(data_folder, paths)
            if args.limit >= 0:
                total_effective = min(total_rows, args.limit)
            else:
                total_effective = total_rows
            row_start = (total_effective * RANK) // TOTAL_SPLIT
            row_end = (total_effective * (RANK + 1)) // TOTAL_SPLIT
            data_reader = ParquetReader(
                data_folder=config.data_path,
                file_progress=False,
                doc_progress=False,
                batch_size=args.batch_size,
                limit=-1,
                skip=0,
            )
            doc_iter = _iter_parquet_rank_row_shard(
                data_folder, paths, row_start, row_end, data_reader
            )
        else:
            data_reader = ParquetReader(
                data_folder=config.data_path,
                file_progress=True,
                batch_size=args.batch_size,
                limit=args.limit,
            )
            doc_iter = data_reader.run(rank=0, world_size=1)
    elif args.data_format == "jsonl.gz":
        data_reader = JsonlReader(
            data_folder=config.data_path,
            file_progress=True,
            adapter=get_adapter_func(args.dataset_name),
            limit=args.limit,
        )
        doc_iter = data_reader.run(rank=RANK, world_size=TOTAL_SPLIT)
    else:
        raise ValueError(f"Unknown data_format {args.data_format}")

    rank_doc_total = None
    if args.data_format == "parquet" and paths and data_folder is not None:
        print(
            f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] counting rows in parquet (metadata scan) …",
            flush=True,
        )
        if parquet_row_shard:
            rank_doc_total = max(0, row_end - row_start)
        else:
            rank_doc_total = _parquet_total_rows(data_folder, paths)
            if args.limit >= 0:
                rank_doc_total = min(rank_doc_total, args.limit)
        print(
            f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] parquet rows for this rank: {rank_doc_total}",
            flush=True,
        )

    dir_path = os.path.join(config.save_path, config.save_name + f"_{RANK}")
    os.makedirs(dir_path, exist_ok=True)
    base_name = config.save_name
    stream_window = _stream_window_size(config, args)
    if args.limit > 0:
        stream_window = min(stream_window, args.limit)
    split_sz = _split_size(config)
    print(
        f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] stream_window={stream_window} "
        f"(docs buffered before each vLLM batch); save_interval split_sz={split_sz}",
        flush=True,
    )

    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=256)
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        free_g, total_g = free_b / (1024**3), total_b / (1024**3)
        low = free_g < 4.0
        extra = (
            " WARNING: almost no free VRAM — another process is likely using this GPU. "
            "Run `nvidia-smi`, kill stray `python` PIDs (or use a free GPU); "
            "lowering GPU_MEM only changes this job's pool size and cannot free memory held by others."
            if low
            else ""
        )
        print(
            f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] CUDA mem before vLLM: "
            f"{free_g:.2f} GiB free / {total_g:.2f} GiB total.{extra}",
            flush=True,
        )
    _mml = getattr(args, "max_model_len", 0) or 0
    print(
        f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] loading vLLM ({args.model_path}) — "
        f"gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"enforce_eager={args.enforce_eager}, max_num_seqs={args.max_num_seqs}"
        f"{', max_model_len=' + str(_mml) if _mml > 0 else ''}. "
        f"If OOM during init: free the GPU (`nvidia-smi` → kill other jobs), then if needed lower GPU_MEM / use ENFORCE_EAGER=1. "
        f"This step often takes several minutes …",
        flush=True,
    )
    llm_kwargs = dict(
        model=args.model_path,
        tokenizer_mode="slow" if tokenizer_slow_fallback else "auto",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )
    mml = getattr(args, "max_model_len", 0) or 0
    if mml > 0:
        llm_kwargs["max_model_len"] = mml
    engine = LLM(**llm_kwargs)
    print(
        f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] vLLM ready; entering read/generate loop",
        flush=True,
    )
    t_infer0 = time.perf_counter()
    _progress_last_milestone = 0
    show_progress = not getattr(args, "disable_tqdm", False)

    def _bump_chunk_doc_progress(docs_done: int) -> None:
        nonlocal _progress_last_milestone
        if not show_progress:
            return
        while docs_done >= _progress_last_milestone + 100:
            _progress_last_milestone += 100
            elapsed = max(time.perf_counter() - t_infer0, 1e-9)
            rate = _progress_last_milestone / elapsed
            prefix = f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] finished {_progress_last_milestone} documents"
            if rank_doc_total is not None and rank_doc_total > 0:
                est_total_s = elapsed * (rank_doc_total / _progress_last_milestone)
                remaining = max(0, rank_doc_total - _progress_last_milestone)
                eta_s = remaining / rate if rate > 0 else float("inf")
                print(
                    f"{prefix} (~{rank_doc_total} on this rank); "
                    f"{rate:.3f} doc/s; ETA this rank ~{eta_s / 3600:.2f} h; "
                    f"extrapolated wall this rank ~{est_total_s / 3600:.2f} h",
                    flush=True,
                )
            else:
                print(
                    f"{prefix}; {rate:.3f} doc/s (rank total unknown, no ETA)",
                    flush=True,
                )

    stats = {
        "rank": RANK,
        "total_split": TOTAL_SPLIT,
        "documents": 0,
        "pairs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_lines": 0,
        "output_lines": 0,
        "pairs_with_empty_target": 0,
        "parquet_row_shard": parquet_row_shard,
    }
    file_seq = 0
    window = []
    docs_read = 0
    read_heartbeat = max(5000, min(50_000, stream_window // 10 or 5000))

    stat_keys = (
        "documents",
        "pairs",
        "input_tokens",
        "output_tokens",
        "input_lines",
        "output_lines",
        "pairs_with_empty_target",
    )

    for doc in _tqdm_if(show_progress, doc_iter, desc="reading documents"):
        window.append({"text": doc.text, "metadata": doc.metadata})
        docs_read += 1
        if show_progress and docs_read % read_heartbeat == 0:
            print(
                f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] read {docs_read} docs from input; "
                f"window {len(window)}/{stream_window} until next vLLM batch",
                flush=True,
            )
        if len(window) >= stream_window:
            file_seq, delta = _process_pair_batches(
                window,
                engine=engine,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                args=args,
                config=config,
                dir_path=dir_path,
                base_name=base_name,
                file_seq_start=file_seq,
                split_sz=split_sz,
                show_progress=show_progress,
            )
            for k in stat_keys:
                stats[k] += delta[k]
            _bump_chunk_doc_progress(stats["documents"])
            window = []
            if show_progress:
                print(
                    f"[chunk-pair rank {RANK}/{TOTAL_SPLIT}] finished vLLM batch; "
                    f"cumulative pairs this rank: {stats['pairs']}",
                    flush=True,
                )

    if window:
        file_seq, delta = _process_pair_batches(
            window,
            engine=engine,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            args=args,
            config=config,
            dir_path=dir_path,
            base_name=base_name,
            file_seq_start=file_seq,
            split_sz=split_sz,
            show_progress=show_progress,
        )
        for k in stat_keys:
            stats[k] += delta[k]
        _bump_chunk_doc_progress(stats["documents"])

    t_end = time.perf_counter()
    stats["wall_seconds_total"] = t_end - t_wall0
    stats["wall_seconds_after_model_load"] = t_end - t_infer0
    stats["model_path"] = args.model_path
    stats["data_path"] = config.data_path
    stats["stream_window"] = stream_window
    if parquet_row_shard:
        stats["parquet_total_rows"] = total_rows
        stats["parquet_total_effective"] = total_effective
        stats["parquet_row_start"] = row_start
        stats["parquet_row_end"] = row_end

    tok_rm = max(0, stats["input_tokens"] - stats["output_tokens"])
    line_rm = max(0, stats["input_lines"] - stats["output_lines"])
    stats["tokens_removed_net"] = tok_rm
    stats["lines_removed_net"] = line_rm

    if args.stats_json:
        stats_path = args.stats_json
    else:
        os.makedirs(config.save_path, exist_ok=True)
        stats_path = os.path.join(
            config.save_path, f"chunk_pair_export_stats_rank{RANK}.json"
        )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"\n=== chunk pair export (rank {RANK} / total_split {TOTAL_SPLIT}) ===\n"
        f"documents processed: {stats['documents']}\n"
        f"pair rows written: {stats['pairs']}\n"
        f"pairs with empty target: {stats['pairs_with_empty_target']}\n"
        f"total input tokens (doc text): {stats['input_tokens']}\n"
        f"total output tokens (chunk targets): {stats['output_tokens']}\n"
        f"tokens removed (max(0, in - out)): {tok_rm}\n"
        f"total input lines: {stats['input_lines']}\n"
        f"total output lines (targets): {stats['output_lines']}\n"
        f"lines removed (max(0, in - out)): {line_rm}\n"
        f"wall time total (s): {stats['wall_seconds_total']:.3f}\n"
        f"wall time after model load (s): {stats['wall_seconds_after_model_load']:.3f}\n"
        f"stats json: {stats_path}\n",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Parquet + multi-rank: global max documents across all ranks. "
        "Otherwise jsonl / single-rank parquet: reader limit (-1 = no cap).",
    )
    parser.add_argument("--data_format", type=str, default="parquet")
    parser.add_argument("--threshold_1", type=float, default=0.0)
    parser.add_argument("--threshold_2", type=float, default=0.95)
    parser.add_argument("--error_op", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="redpajama-v2")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.72")),
        help="vLLM fraction of GPU memory for weights+KV pool (lower if OOM at init).",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable vLLM CUDA graphs (less peak VRAM; often fixes last-block OOM).",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
        help="vLLM max concurrent sequences (lower may reduce memory).",
    )
    parser.add_argument(
        "--stream_window",
        type=int,
        default=0,
        help="Max documents to buffer per flush (0 = auto from save_interval, capped at 100k).",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default="",
        help="Write per-rank stats JSON to this path (default: under save_path).",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm bars (less Python overhead on huge runs).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=0,
        help="If >0, pass to vLLM LLM(max_model_len=...); reduces prompt truncation warnings.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Override config save_path (e.g. benchmark temp dir).",
    )
    args = parser.parse_args()
    main(args)
