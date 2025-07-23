#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=lb-mel
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_24GB'

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection


# --lebench_subset xlg, lg, md, md_clean, sm

python run_get_mfcc_lebenchmark.py hparams/lebench_data.yaml \
    --csv_location /users/rwhetten/LeBenchmark \
    --lebench_subset sm \
    --save_int 60 \
    --ckpt_path ckpt.pkl \
    --feature_function_name mel



# python run_get_mfcc_lebenchmark.py hparams/lebench_data.yaml \
#     --csv_location /users/rwhetten/LeBenchmark \
#     --lebench_subset xlg \
#     --save_int 60 \
#     --ckpt_path ckpt.pkl \
#     --feature_function_name mel