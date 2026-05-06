import argparse
import json
import os
import random
import time

from datasets import Dataset
from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_utils import get_adapter_func, split_into_batches
from utils.chunk_utils import execute_meta_operations
from vllm import LLM, SamplingParams

from data_gen.configs import GentaskConfig

random.seed(42)


# dummy env constants for multi-gpu & multi-node
NODE_GPUS = int(os.environ.get("NODE_GPUS", 8))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))


def _cuda_visible_device_index() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    first = raw.split(",")[0].strip()
    if first.lower() in ("", "none"):
        return 0
    return int(first)


CUDA_DEVICE = _cuda_visible_device_index()
TOTAL_SPLIT = int(os.environ["TOTAL_SPLIT"])
RANK = CUDA_DEVICE + NODE_RANK * NODE_GPUS


def trunc_text(text: str, tokenizer, max_token=1500, max_digits=3):
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for idx_line, line in enumerate(lines):
        normalize_line = f"[{idx_line:0{max_digits}d}]{line}"
        line_token_count = len(tokenizer.encode(normalize_line))

        # if cur line can be appended in current chunk
        if current_chunk_token_count + line_token_count <= max_token:
            current_chunk.append(normalize_line)
            current_chunk_token_count += line_token_count
        # if cur line cannot be appended in current chunk
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [normalize_line]
            current_chunk_token_count = line_token_count
            if line_token_count > max_token:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_chunk_token_count = 0

    if len(current_chunk) > 0:
        chunks.append("\n".join(current_chunk))

    return chunks


def merge_chunks(chunk_list, chunk_lengths):
    merged_documents = []
    current_index = 0

    for length in chunk_lengths:
        document_chunks = chunk_list[current_index : current_index + length]
        merged_document = "\n".join(document_chunks)
        merged_documents.append(merged_document)
        current_index += length

    return merged_documents


def _doc_input_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def _stream_window_size(config, args) -> int:
    if args.stream_window > 0:
        return args.stream_window
    si = config.save_interval
    if si and si > 0:
        return min(si, 100_000)
    return 50_000


def _split_size(config) -> int:
    si = config.save_interval
    if si and si > 0:
        return si
    return 10**12


def _sorted_parquet_paths(data_folder) -> list[str]:
    paths = [
        p
        for p in data_folder.list_files(recursive=True)
        if p.endswith(".parquet")
    ]
    return sorted(paths)


def _parquet_total_rows(data_folder, paths: list[str]) -> int:
    import pyarrow.parquet as pq

    total = 0
    for filepath in paths:
        with data_folder.open(filepath, "rb") as f:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
    return total


def _iter_parquet_rank_row_shard(
    data_folder,
    paths: list[str],
    row_lo: int,
    row_hi: int,
    reader: ParquetReader,
):
    """Yield Documents for global row indices in [row_lo, row_hi) using row-group reads."""
    import pyarrow.parquet as pq

    if row_lo >= row_hi:
        return
    global_i = 0
    for filepath in paths:
        with data_folder.open(filepath, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                g0 = global_i
                g1 = global_i + n
                if g1 <= row_lo:
                    global_i = g1
                    continue
                if g0 >= row_hi:
                    return
                lo_local = max(0, row_lo - g0)
                hi_local = min(n, row_hi - g0)
                if lo_local < hi_local:
                    tbl = pf.read_row_group(rg)
                    sub = tbl.slice(lo_local, hi_local - lo_local)
                    for j, line in enumerate(sub.to_pylist()):
                        row_dict = dict(line)
                        gid = g0 + lo_local + j
                        doc = reader.get_document_from_dict(
                            row_dict, filepath, str(gid)
                        )
                        if doc is None:
                            yield Document(text="", id=str(gid), metadata={})
                        else:
                            yield doc
                global_i = g1


def _process_batches(
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
):
    """Run chunk LM + parquet writes for documents in ``arguments``. Returns (next_file_seq, stats_delta)."""
    batches = split_into_batches(arguments, split_sz)
    file_seq = file_seq_start
    d_docs = 0
    d_in_tok = 0
    d_out_tok = 0
    d_in_lines = 0
    d_out_lines = 0
    d_docs_dropped = 0

    for i, batch in enumerate(tqdm(batches, desc="save_interval batches")):
        rets = []
        chunk_lens = []
        for sample in tqdm(batch, total=len(batch), unit="tokenizing", leave=False):
            raw = sample.get("text") or ""
            d_in_tok += _doc_input_tokens(tokenizer, raw)
            d_in_lines += len(raw.splitlines())

            if sample["text"] == "":
                chunk_lens.append(0)
                continue
            user_msgs = trunc_text(sample["text"], tokenizer, max_token=1500)
            chunk_lens.append(len(user_msgs))
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

                rets.append(total_msg)
        if not rets:
            merged_outputs = merge_chunks([], chunk_lens)
        else:
            outputs = engine.generate(rets, sampling_params)
            outputs = [item.outputs[0].text.strip(" ") for item in outputs]
            merged_outputs = merge_chunks(outputs, chunk_lens)
        rets = []
        for sample, output in zip(batch, merged_outputs):
            meta = sample.get("metadata") or {}
            raw_content = meta.get("raw_content") or sample.get("text") or ""
            doc_program = meta.get("doc_program", "")
            refined = execute_meta_operations(
                text=sample["text"],
                operations=output,
                threshold_1=args.threshold_1,
                threshold_2=args.threshold_2,
                error_op=args.error_op,
            )
            d_docs += 1
            ot = _doc_input_tokens(tokenizer, refined) if refined else 0
            d_out_tok += ot
            ol = len(refined.splitlines()) if refined else 0
            d_out_lines += ol
            if not refined.strip():
                d_docs_dropped += 1
            rets.append(
                {
                    "raw_content": raw_content,
                    "doc_content": sample["text"],
                    "text": refined,
                    "metadata": {
                        "doc_program": doc_program,
                        "chunk_program": output,
                    },
                }
            )

        intermediate_ds = Dataset.from_list(rets)
        file_seq += 1
        out_path = os.path.join(dir_path, f"{base_name}_{file_seq:06d}.parquet")
        intermediate_ds.to_parquet(out_path)

    return file_seq, {
        "documents": d_docs,
        "input_tokens": d_in_tok,
        "output_tokens": d_out_tok,
        "input_lines": d_in_lines,
        "output_lines": d_out_lines,
        "documents_dropped_empty": d_docs_dropped,
    }


def main(args):
    t_wall0 = time.perf_counter()
    config = GentaskConfig().from_yaml(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data_reader = None
    doc_iter = None
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

    dir_path = os.path.join(config.save_path, config.save_name + f"_{RANK}")
    os.makedirs(dir_path, exist_ok=True)
    base_name = config.save_name
    stream_window = _stream_window_size(config, args)
    split_sz = _split_size(config)

    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=256)
    engine = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    t_infer0 = time.perf_counter()

    stats = {
        "rank": RANK,
        "total_split": TOTAL_SPLIT,
        "documents": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_lines": 0,
        "output_lines": 0,
        "documents_dropped_empty": 0,
        "parquet_row_shard": parquet_row_shard,
    }
    file_seq = 0
    window = []

    for doc in tqdm(doc_iter, desc="reading documents"):
        window.append({"text": doc.text, "metadata": doc.metadata})
        if len(window) >= stream_window:
            file_seq, delta = _process_batches(
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
            )
            for k in ("documents", "input_tokens", "output_tokens", "input_lines", "output_lines", "documents_dropped_empty"):
                stats[k] += delta[k]
            window = []

    if window:
        file_seq, delta = _process_batches(
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
        )
        for k in ("documents", "input_tokens", "output_tokens", "input_lines", "output_lines", "documents_dropped_empty"):
            stats[k] += delta[k]

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
            config.save_path, f"chunk_refining_stats_rank{RANK}.json"
        )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"\n=== chunk refining statistics (rank {RANK} / total_split {TOTAL_SPLIT}) ===\n"
        f"documents processed: {stats['documents']}\n"
        f"total input tokens (doc text): {stats['input_tokens']}\n"
        f"total output tokens (refined text): {stats['output_tokens']}\n"
        f"tokens removed (max(0, in - out)): {tok_rm}\n"
        f"total input lines: {stats['input_lines']}\n"
        f"total output lines: {stats['output_lines']}\n"
        f"lines removed (max(0, in - out)): {line_rm}\n"
        f"documents with empty refined output: {stats['documents_dropped_empty']}\n"
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
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        help="vLLM KV cache fraction (lower if CUDA OOM on busy GPUs).",
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
    args = parser.parse_args()
    main(args)
