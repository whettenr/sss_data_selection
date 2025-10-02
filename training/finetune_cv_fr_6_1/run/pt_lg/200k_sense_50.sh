#!/bin/bash

#SBATCH --job-name=sen200k
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=13:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ft%j.log
#SBATCH --mail-user=ryan.whetten@univ-avignon.fr
#SBATCH --mail-type=ALL


module unload pytorch-gpu
module load arch/a100
module load pytorch-gpu/py3/2.6.0
conda activate dataselection
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1

train=loq/train_with_BEST-RQ.py
hyparams=loq/train_loq_with_BEST-RQ.yaml
save_name=sense_50
step=200
hub=/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/results/loq_100M_large_${save_name}/steps/CKPT+step_${step}000

tls_subset=small
hf_hub=$DSDIR/HuggingFace/speechbrain/LoquaciousSet
hf_caching_dir=$SCRATCH/HuggingFace/speechbrain/LoquaciousSet
train_csv=loq/loquacious_small_train.csv

output_folder=results/loq_100M_${save_name}_${tls_subset}_ptlg/step_${step}k_bpe


python $train $hyparams \
    --output_folder $output_folder \
    --hub $hub \
    --train_csv $train_csv \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --output_neurons_ctc 1024 \
    --token_type bpe \
    --pt_model_output_dim 640 \
    --max_batch_length_train 1000