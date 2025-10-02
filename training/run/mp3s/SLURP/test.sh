#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --job-name=ls_brqlg
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=BEGIN,END,FAIL


cd /local_disk/apollon/rwhetten/sss_data_selection/training
conda activate aa

# model settings
num_layers='13' # output of CNN + 12 layers
num_encoder_layers='12'
encoder_dim='640'
attention_type=RoPEMHA

# ckpt settings
pt_ds=large
feat_name=base
hub=results/jz/loq_100M_${pt_ds}_${feat_name}/CKPT+step_200000
output_folder=results/MP3S/loq_100M_${pt_ds}_${feat_name}

# benchmark settings
benchmark_base='/users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S'
csv_location=/users/rwhetten/SLURP/csv
DatasetsFolders=('/users/rwhetten/SLURP' '/users/SLURP')
ConsideredTasks=('SLURP' 'SLURP')
DownStreams=('LSTM_linear' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_base/$task/$downstream/train.py $benchmark_base/$task/$downstream/hparams/ssl_brq_kernel_5.yaml \
        --num_layers_ssl $num_layers \
        --num_encoder_layers $num_encoder_layers \
        --ssl_hub $hub \
        --encoder_dim $encoder_dim \
        --output_folder $output_folder/$task/$downstream \
        --attention_type $attention_type \
        --csv_location $csv_location \
        --data_folder $dataset_folder
done


