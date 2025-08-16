#!/bin/bash

#SBATCH --job-name=loq
#SBATCH -C a100
#SBATCH --account=dha@a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ft%j.log
# SBATCH --mail-user=ryan.whetten@univ-avignon.fr
# SBATCH --mail-type=ALL



# module load arch/a100
# module load pytorch-gpu/py3/2.1.1
# conda activate ft-sb
# cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/

# train=loq_train.py
# hparams=hparams/loq_BEST-RQ_250M.yaml

# lr=0.0005
# feat_name=base
# tls_subset=medium
# output_folder=results/loq_250M_${tls_subset}_${feat_name}
# train_csv=/local_disk/apollon/rwhetten/loquacious_${tls_subset}_train.csv

# torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=8 $train $hparams --find_unused_parameters \
#     --grad_accumulation_factor 4 \
#     --output_folder $output_folder \
#     --train_csv $train_csv \
#     --valid_csv $valid_csv \
#     --skip_prep true \
#     --lr $lr \
#     --optimizer_step_limit 220000 \
#     --tls_subset $tls_subset \
#     --hf_hub $hf_hub \
#     --hf_caching_dir $hf_caching_dir \
#     --filter false \
#     --hf_caching_dir /local_disk/apollon/rwhetten/hf_root/datasets



module load arch/a100
module load pytorch-gpu/py3/2.1.1
conda activate ft-sb
cd /lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/

train=loq_train.py
hparams=hparams/loq_BEST-RQ_250M.yaml

lr=0.0005
feat_name=base
tls_subset=medium
output_folder=results/loq_250M_${tls_subset}_${feat_name}
train_csv=/local_disk/apollon/rwhetten/loquacious_${tls_subset}_train.csv

hf_hub=$DSDIR/HuggingFace/speechbrain/LoquaciousSet
hf_caching_dir=$SCRATCH/HuggingFace/speechbrain/LoquaciousSet

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=2 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 4 \
    --output_folder $output_folder \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --skip_prep true \
    --lr $lr \
    --optimizer_step_limit 220000 \
    --tls_subset $tls_subset \
    --hf_hub $hf_hub \
    --hf_caching_dir $hf_caching_dir \
    --filter false