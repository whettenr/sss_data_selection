#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --job-name=sm_sense
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_32GB&GPURAM_Max_32GB'

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection

# test
python run_get_sense.py sense/data_select.yaml \
    --tls_subset small \
    --hf_hub speechbrain/LoquaciousSet \
    --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets \
    --save_int 1 \
    --save_dir features \
    --feature_function_name sense \
    --sense_location /data/coros3/smdhaffar/SENSE/CKPT+checkpoint_epoch10/ \
    --output_folder /local_disk/apollon/rwhetten/sss_data_selection/sense/pt_store \
    --num_workers 4 \
    --profiler false \
    --max_batch_length_train 600 \
    --batch_ordering descending \
    --ckpt_path p1