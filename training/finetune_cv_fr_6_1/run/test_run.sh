#!/bin/bash

#SBATCH --job-name=flb   # nom du job
#SBATCH --account=nkp@v100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=30:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ft%j.log  # log file


module load pytorch-gpu/py3/2.0.1
conda activate ft-sb
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1

train=train_with_BEST-RQ.py
hyparams=train_fr_with_BEST-RQ.yaml
hub=results/lebench_sm_base/steps/CKPT+step_50000/

data_folder=/lustre/fsmisc/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr
csv_folder=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs
output_folder=results/ft/base
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hyparams \
    --output_folder results/ft/base \
    --hub $hub \
    --data_folder $data_folder \
    --train_csv $csv_folder/train.csv \
    --valid_csv $csv_folder/dev.csv \
    --test_csv $csv_folder/test.csv \
    --skip_prep True
