#!/bin/bash
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
csv_location=/users/rwhetten/ICASSP_2024/best-rq-test/results

DatasetsFolders=('/corpus/LibriSpeech/' '/corpus/LibriSpeech/')
ConsideredTasks=('LibriSpeech' 'LibriSpeech')
DownStreams=('LSTM' 'contextnet')

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
        --data_folder $dataset_folder # --test_only --language_modelling True
done




# Tested for LibriSpeech
# change kernel size
# remove quantizer and linear layer from pt_model


# testing no checkpoint
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
output_folder=results/MP3S/loq_100M_${pt_ds}_${feat_name}_test_nocheckpoint_checkweights

# benchmark settings
benchmark_base='/users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S'
csv_location=/users/rwhetten/ICASSP_2024/best-rq-test/results

DatasetsFolders=('/corpus/LibriSpeech/')
ConsideredTasks=('LibriSpeech')
DownStreams=('LSTM')

task=${ConsideredTasks[0]}
downstream=${DownStreams[0]}
dataset_folder=${DatasetsFolders[0]}
python $benchmark_base/$task/$downstream/train_test_nockpt.py $benchmark_base/$task/$downstream/hparams/ssl_brq_kernel_5.yaml \
        --num_layers_ssl $num_layers \
        --num_encoder_layers $num_encoder_layers \
        --ssl_hub $hub \
        --encoder_dim $encoder_dim \
        --output_folder $output_folder/$task/$downstream \
        --attention_type $attention_type \
        --csv_location $csv_location \
        --data_folder $dataset_folder







cd /local_disk/apollon/rwhetten/sss_data_selection/training
conda activate aa

# model settings
num_layers='13' # output of CNN + 12 layers
num_encoder_layers='12'
encoder_dim='640'
attention_type=RoPEMHA

# ckpt settings
pt_ds=large
feat_name=length_50
hub=results/jz/loq_100M_${pt_ds}_${feat_name}/CKPT+step_200000
output_folder=results/MP3S/loq_100M_${pt_ds}_${feat_name}_updatesb

# benchmark settings
benchmark_base='/users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S'
csv_location=/users/rwhetten/ICASSP_2024/best-rq-test/results

DatasetsFolders=('/corpus/LibriSpeech/' '/corpus/LibriSpeech/')
ConsideredTasks=('LibriSpeech' 'LibriSpeech')
DownStreams=('LSTM' 'contextnet')

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





conda activate aa

cd /local_disk/apollon/rwhetten/sss_data_selection/training
hub=/local_disk/apollon/rwhetten/sss_data_selection/training/results/jz/loq_100M_large_base/CKPT+step_200000
encoder_dim='640'

train=/users/rwhetten/attention_alt/brq-att-alt-exp/finetune/ft_brq.py
hparams=/users/rwhetten/attention_alt/brq-att-alt-exp/finetune/ft_brq_kernel_5.yaml

python $train $hparams \
    --data_folder /gpfsdswork/dataset/LibriSpeechAsrCorpus \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type RoPEMHA \
    --output_folder results/ft_test_base