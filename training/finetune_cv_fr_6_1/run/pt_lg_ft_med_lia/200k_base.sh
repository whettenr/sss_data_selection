#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_24GB'
#SBATCH --mail-type=END,FAIL



conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection/training/finetune_cv_fr_6_1

train=loq/train_with_BEST-RQ.py
hyparams=loq/train_loq_with_BEST-RQ.yaml
save_name=base
step=200
hub=/local_disk/apollon/rwhetten/sss_data_selection/training/results/jz/loq_100M_large_${save_name}/CKPT+step_${step}000

tls_subset=medium
hf_hub=speechbrain/LoquaciousSet
hf_caching_dir=/local_disk/apollon/rwhetten/hf_root/datasets
train_csv=/local_disk/apollon/rwhetten/loquacious_${tls_subset}_train.csv

output_folder=results/loq_100M_${save_name}_${tls_subset}_ptlg/step_${step}k_bpe


python $train $hyparams \
    --output_folder $output_folder \
    --hub $hub \
    --train_csv $train_csv \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --output_neurons_ctc 5120 \
    --token_type bpe \
    --pt_model_output_dim 640 \
    --max_batch_length_train 300 \
    --grad_accumulation_factor 2 \
    --number_of_epochs 10