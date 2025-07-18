#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=spe
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_16GB'

conda activate aa
cd /local_disk/apollon/rwhetten/dataselection

python run_get_semantic_text.py hparams/data_select.yaml \
    --tls_subset large \
    --hf_hub speechbrain/LoquaciousSet \
    --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets \
    --save_int 60 \
    --ckpt_path ckpt.pkl \
    --feature_function_name text_emb