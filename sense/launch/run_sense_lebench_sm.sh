#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --job-name=lbsm_sense
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_32GB&GPURAM_Max_32GB'

conda activate aa
cd /local_disk/apollon/rwhetten/sss_data_selection

python run_get_sense_lebenchmark.py sense/lebench_data.yaml \
    --csv_location /users/rwhetten/LeBenchmark \
    --lebench_subset sm \
    --save_int 20 \
    --save_dir features \
    --feature_function_name sense \
    --sense_location /data/coros3/smdhaffar/SENSE/CKPT+checkpoint_epoch10/ \
    --output_folder /local_disk/apollon/rwhetten/sss_data_selection/sense/pt_store \
    --num_workers 4 \
    --profiler false \
    --max_batch_length_train 600 \
    --batch_ordering descending \
    --ckpt_path p1