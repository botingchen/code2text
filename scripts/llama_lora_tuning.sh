#!/bin/bash
#SBATCH --job-name=eval-lora-hparam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/eval_lora_llama_%A_%a.out
#SBATCH --error=logs/slurm/eval_lora_llama_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiqunh@andrew.cmu.edu

source ~/.bashrc
conda activate llm
cd /home/yiqunh/code2text

HPARAM_GRID=(
"lora_r=16  lora_alpha=32 lora_dropout=0.05 learning_rate=2e-4 gradient_accumulation_steps=8 per_device_train_batch_size=2 num_train_epochs=1"
)

HPARAMS="${HPARAM_GRID[$SLURM_ARRAY_TASK_ID]}"
for kv in $HPARAMS; do eval "$kv"; done

DATA_DIR=${DATA_DIR:-data/processed/code_x_glue_go_lenfiltered_sampled}

RUN_NAME="r${lora_r}_a${lora_alpha}_lr${learning_rate}_bs${per_device_train_batch_size}_ga${gradient_accumulation_steps}_ep${num_train_epochs}"

srun python3 evaluation/eval_lora.py \
  --data-dir data/processed/code_x_glue_go_lenfiltered_sampled \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --lora-output-dir outputs/lora/llama \
  --max-seq-length 1024 \
  --max-new-tokens 64 \
  --per-device-train-batch-size "${per_device_train_batch_size}" \
  --gradient-accumulation-steps "${gradient_accumulation_steps}" \
  --learning-rate "${learning_rate}" \
  --logging-steps 25 \
  --lora-r "${lora_r}" \
  --lora-alpha "${lora_alpha}" \
  --lora-dropout "${lora_dropout}" \
  --num-train-epochs "${num_train_epochs}"