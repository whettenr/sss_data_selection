#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL



conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection/training/finetune_cv_fr_6_1

train=loq/train_with_BEST-RQ.py
hyparams=loq/train_loq_with_BEST-RQ.yaml
hub=/local_disk/apollon/rwhetten/sss_data_selection/training/results/loq_small_base_50/steps/CKPT+step_100000

output_folder=results/test_bpe_1024_ld
tls_subset=small
hf_hub=speechbrain/LoquaciousSet
hf_caching_dir=/local_disk/apollon/rwhetten/hf_root/datasets
train_csv=/local_disk/apollon/rwhetten/loquacious_small_train.csv

python $train $hyparams \
    --output_folder $output_folder \
    --hub $hub \
    --train_csv $train_csv \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --sorting ascending \
    --output_neurons_ctc 1024 \
    --token_type bpe