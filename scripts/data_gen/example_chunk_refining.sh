#!/bin/bash
# Example Slurm launcher for chunk-level refining (see README "Large Scale Data Refining").
# Fill in #SBATCH lines and site setup_* / $TINYLM_* paths. Uses conda env: refining.
# Point --config_path at a YAML whose data_path/save_path match your corpus (e.g. fineweb:
# data_gen/configs/apply_chunk_refining_fineweb_first_parquet.yaml). Non-Slurm helper:
# scripts/data_gen/run_chunk_refining_fineweb_first_parquet.sh
#SBATCH --job-name=prox_chunk_refining_xs
#SBATCH --output=<expected_output_file>
#SBATCH --partition=<your_partition>
#SBATCH --error=<expected_error_file>
#SBATCH --time=50:00:00
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

# setup env 
chmod +x setup_personal_env.sh
chmod +x setup_common_env.sh
source setup_personal_env.sh
source setup_common_env.sh

# activate conda env
source $TINYLM_CONDA_DIR
conda activate refining

# enter working dir
cd $TINYLM_WORK_DIR

export NNODE=8
export NGPU=8
export TOTAL_SPLIT=$((NNODE*NGPU))

cmd="
for i in \$(seq 0 \$((NGPU-1))); do
    TOTAL_SPLIT=$TOTAL_SPLIT \\
    NODE_GPUS=$NGPU \\
    NODE_RANK=\$SLURM_NODEID \\
    CUDA_VISIBLE_DEVICES=\$i \\
    python -m data_gen.tasks.apply_chunk_refining \\
        --data_format parquet \\
        --limit -1 \\
        --model_path gair-prox/web-chunk-refining-lm \\
        --config_path data_gen/configs/apply_chunk_refining.yaml \\
        > ./logging/apply_chunk_refining_\${SLURM_NODEID}_\${i}.log 2>&1 &
done
wait
"

echo "Executing command:"
echo "$cmd"

srun bash -c "$cmd"

# # ****************************************************
# # scripts for single node: (debug)
# # ****************************************************
# # setup env
# chmod +x setup_personal_env.sh
# chmod +x setup_common_env.sh
# source setup_personal_env.sh
# source setup_common_env.sh

# # activate conda env
# source $TINYLM_CONDA_DIR
# conda activate llama_factory

# # enter working dir
# cd $TINYLM_WORK_DIR

# export NNODE=1
# export NGPU=1
# # total split (int) = nnode * ngpu, write in shell expression
# export TOTAL_SPLIT=$((NNODE*NGPU))
# export SLURM_NODEID=0
# for i in $(seq 0 $((NGPU-1))); do
#   TOTAL_SPLIT=$TOTAL_SPLIT NODE_GPUS=$NGPU NODE_RANK=$SLURM_NODEID CUDA_VISIBLE_DEVICES=$i \
#   python -m data_gen.tasks.apply_chunk_refining \
#     --data_format parquet \
#     --limit 1000 \
#     --model_path gair-prox/chunk_refining_web_lm \
#     --config_path data_gen/configs/apply_chunk_refining.yaml \
#     > ./logging/apply_chunk_refining_${SLURM_NODEID}_${i}.log &
# done
