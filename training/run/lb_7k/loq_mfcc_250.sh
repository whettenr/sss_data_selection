#!/bin/bash

#SBATCH --job-name=loqMF
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ft%j.log
# SBATCH --mail-user=ryan.whetten@univ-avignon.fr
# SBATCH --mail-type=ALL

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection/training

train=loq_train.py
hparams=hparams/loq_BEST-RQ_250M.yaml

lr=0.0005
feat_name=mfcc
tls_subset=medium
output_folder=results/loq_250M_${tls_subset}_${feat_name}_50
train_csv=/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_${tls_subset}/${feat_name}_0.5.csv

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=8 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 4 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 150000 \
    --tls_subset $tls_subset \
    --hf_hub speechbrain/LoquaciousSet \
    --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets