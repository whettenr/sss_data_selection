#!/bin/bash

#SBATCH --job-name=loqb100k
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
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1

train=loq/train_with_BEST-RQ.py
hyparams=loq/train_loq_with_BEST-RQ.yaml
save_name=base
step=100
hub=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/results/loq_small_${save_name}/steps/CKPT+step_${step}000

output_folder=results/loq_${save_name}/step_${step}k_bpe
tls_subset=small
hf_hub=$DSDIR/HuggingFace/speechbrain/LoquaciousSet
hf_caching_dir=$SCRATCH/HuggingFace/speechbrain/LoquaciousSet
train_csv=loq/loquacious_small_train.csv



python $train $hyparams \
    --output_folder $output_folder \
    --hub $hub \
    --train_csv $train_csv \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --sorting ascending \
    --output_neurons_ctc 1024 \
    --token_type bpe