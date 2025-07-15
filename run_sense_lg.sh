#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --job-name=lg_sense
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_32GB'

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection


python run_get_sense.py sense/data_select.yaml \
    --tls_subset large \
    --hf_hub speechbrain/LoquaciousSet \
    --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets \
    --save_int 60 \
    --ckpt_path ckpt.pkl \
    --feature_function_name sense \
    --sense_location /data/coros3/smdhaffar/SENSE/CKPT+checkpoint_epoch10/ \
    --output_folder /local_disk/apollon/rwhetten/sss_data_selection/sense/pt_store \
    --max_batch_length_train 600