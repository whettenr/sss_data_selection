#!/bin/bash

#SBATCH --job-name=lb
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/lb100M_%j.log
#SBATCH --mail-user=ryan.whetten@univ-avignon.fr
#SBATCH --mail-type=ALL



module unload pytorch-gpu
module load arch/a100
module load pytorch-gpu/py3/2.6.0

conda activate dataselection
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/

train=lebench_train.py
hparams=hparams/lebench_BEST-RQ_100M.yaml

lr=0.0005
subset=lg
feat_name=base
output_folder=results/lb_100M_${subset}_${feat_name}

train_csv=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/lb_${subset}/jz_train_fix.csv
valid_csv=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/lb_${subset}/jz_mls_french-dev.csv

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=1 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 1 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 200000 \
    --precision bf16 \
    --seconds_per_batch 200 \
    --ckpt_interval_minutes 15

