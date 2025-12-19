#!/bin/bash

#SBATCH --job-name=lbsensenld
#SBATCH -C a100
#SBATCH --account=oou@a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/lb330M_sense_%j.log
#SBATCH --mail-user=ryan.whetten@univ-avignon.fr
#SBATCH --mail-type=ALL



module unload pytorch-gpu
module load arch/a100
module load pytorch-gpu/py3/2.6.0

conda activate dataselection
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/

train=lebench_train.py
hparams=hparams/lebench_BEST-RQ_330M.yaml

lr=0.0002
subset=med
feat_name=sense
output_folder=results/lb_330M_no_layerdrop_${subset}_${feat_name}

train_csv=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/lb_${subset}/jz_${feat_name}_0.5.csv
valid_csv=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/lb_${subset}/jz_mls_french-dev.csv

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=8 $train $hparams \
    --grad_accumulation_factor 2 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 200000 \
    --precision bf16 \
    --encoder_layerdrop 0.0