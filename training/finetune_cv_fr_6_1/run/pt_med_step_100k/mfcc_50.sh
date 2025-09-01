#!/bin/bash

#SBATCH --job-name=cvmf50
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ft%j.log
# SBATCH --mail-user=ryan.whetten@univ-avignon.fr
# SBATCH --mail-type=ALL


module load arch/a100
module load pytorch-gpu/py3/2.1.1
conda activate ft-sb
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1/

train=train_with_BEST-RQ.py
hyparams=train_fr_with_BEST-RQ.yaml
save_name=mfcc_50
step=100
hub=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/results/lebench_sm_${save_name}/steps/CKPT+step_${step}000

data_folder=/lustre/fsmisc/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr
csv_folder=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs
output_folder=results/${save_name}/step_${step}k
batch_size=64
test_batch_size=24


python $train $hyparams \
    --output_folder $output_folder \
    --hub $hub \
    --data_folder $data_folder \
    --train_csv $csv_folder/train.csv \
    --valid_csv $csv_folder/dev.csv \
    --test_csv $csv_folder/test.csv \
    --skip_prep True \
    --batch_size $batch_size \
    --test_batch_size $test_batch_size \
    --test_only
    # --kenlm_model_path /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1/fr_5gram.arpa \
    --test_only
