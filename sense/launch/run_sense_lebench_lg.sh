#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=50:00:00
#SBATCH --job-name=lg_lb_sense
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
# SBATCH --array=1-8


conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection

CKPT_PATH="ckpts/p${SLURM_ARRAY_TASK_ID}"

python run_get_sense_lebenchmark.py sense/lebench_data.yaml \
    --csv_location /users/rwhetten/LeBenchmark \
    --lebench_subset lg \
    --save_int 60 \
    --save_dir features \
    --feature_function_name sense \
    --sense_location /data/coros3/smdhaffar/SENSE/CKPT+checkpoint_epoch10/ \
    --output_folder /local_disk/apollon/rwhetten/sss_data_selection/sense/pt_store \
    --num_workers 6 \
    --profiler false \
    --max_batch_length_train 600 \
    --batch_ordering descending \
    --ckpt_path $CKPT_PATH
