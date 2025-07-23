#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=75:00:00
#SBATCH --job-name=lb-spe-lg
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_32GB&GPURAM_Max_32GB'

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection


python run_get_speaker_lebenchmark.py hparams/lebench_data.yaml \
    --csv_location /users/rwhetten/LeBenchmark \
    --lebench_subset xlg \
    --save_int 60 \
    --ckpt_path ckpt.pkl \
    --feature_function_name speaker \
    --num_workers 6
