#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=lb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection/training

train=lebench_train.py
hparams=hparams/lebench_BEST-RQ.yaml

lr=0.0005
output_folder=results/lebench_test
train_csv=/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/lebench_sm/random_0.5.csv
valid_csv=/users/rwhetten/LeBenchmark/sm/mls_french-dev.csv
# train_csv=/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/lebench_sm/mfcc_0.5.csv
# train_csv=/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/lebench_sm/speaker_0.5.csv

# double check --nproc_per_node=
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=2 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 8 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 150000
