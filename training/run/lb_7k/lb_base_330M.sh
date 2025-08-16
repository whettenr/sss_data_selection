#!/bin/bash

#SBATCH --job-name=lb
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

train=lebench_train.py
hparams=hparams/lebench_BEST-RQ_330M.yaml

lr=0.0005
feat_name=base
output_folder=results/lebench_lg_330M_${feat_name}
train_csv=/users/rwhetten/LeBenchmark/lg/train.csv
valid_csv=/users/rwhetten/LeBenchmark/lg/mls_french-dev.csv

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=8 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 4 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 220000

# increase to 24 layers -> 181.4M
# increase hidden dim to 768 -> 258.0M (good for loq comparison)
# increase hidden dim to 1024 -> 388.0M
# decrease num layers to 20 -> 324.9M
# 768, 20 layer, dffn 3072 -> 279.1M
# d_model: 768 num_encoder_layers: 24, d_ffn: 3072 -> 333.6M
