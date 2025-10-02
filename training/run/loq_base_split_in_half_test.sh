#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --job-name=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection/training

train=loq_train_split_in_half.py
hparams=hparams/loq_BEST-RQ.yaml

lr=0.0005
feat_name=base
tls_subset=small
output_folder=results/loq_${tls_subset}_${feat_name}_split_in_half_test
train_csv=/local_disk/apollon/rwhetten/loquacious_small_train.csv

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 2 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 150000 \
    --tls_subset $tls_subset \
    --hf_hub speechbrain/LoquaciousSet \
    --filter false \
    --num_workers 4 \
    --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets