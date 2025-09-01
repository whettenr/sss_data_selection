#!/bin/bash
#SBATCH --job-name=lmd_sens   # nom du job
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=/lustre/fsn1/projects/rech/nkp/uaj64gk/log/loqsense_100M_%j.log  # log file
#SBATCH --mail-user=ryan.whetten@univ-avignon.fr
#SBATCH --mail-type=ALL

module unload pytorch-gpu
module load arch/a100
module load pytorch-gpu/py3/2.6.0

conda activate dataselection
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/

train=loq_train.py
hparams=hparams/loq_BEST-RQ_100M.yaml

lr=0.0005
feat_name=sense
tls_subset=medium
output_folder=results/loq_100M_${tls_subset}_${feat_name}_50
train_csv=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/loq_csvs/loq_${tls_subset}/${feat_name}_0.5.csv

hf_hub=$DSDIR/HuggingFace/speechbrain/LoquaciousSet
hf_caching_dir=$SCRATCH/HuggingFace/speechbrain/LoquaciousSet


torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=8 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 1 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 200000 \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --max_batch_length_train 800 \
    --precision bf16

    