#!/bin/bash

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model path provided."
        echo "Usage: $0 <model_path>"
        exit 1
fi

OUTPUT_PATH="${MODEL_PATH%/}/lm_eval_results"

accelerate launch --num_processes 1 -m lm_eval \
        --model hf \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True,parallelize=False \
        --tasks wikitext,mmlu \
        --device cuda \
        --batch_size auto \
        --output_path "$OUTPUT_PATH" \
        --log_samples

echo "[eval_lm_eval] results saved under: $OUTPUT_PATH"
