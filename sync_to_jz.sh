source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/results"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"

rsync -azh "$source_path" "$target_path" \
    --progress \
    --exclude 'optimizer.ckpt' \
    --exclude 'quantizer.ckpt' \
    --exclude 'save'


# transfer hparams
source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/hparams"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"
rsync -azh "$source_path" "$target_path" \
    --progress 

# transfer run scripts
source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/run"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"
rsync -azh "$source_path" "$target_path" \
    --progress

# transfer py files
source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"
rsync -azh "$source_path" "$target_path" \
    --progress \
    --include '*.py' \
    --exclude '*'

# transfer csv
source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/csvs/loq_medium"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs"
rsync -azh "$source_path" "$target_path" \
    --progress

source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/csvs/loq_large"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/loq_csvs"
rsync -azh "$source_path" "$target_path" \
    --progress

source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs"
rsync -azh "$source_path" "$target_path" \
    --progress

# source_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1"
# target_path="/local_disk/apollon/rwhetten/sss_data_selection/training/"

# rsync -azh "$source_path" "$target_path" \
#     --progress \
#     --exclude 'optimizer.ckpt' \
#     --exclude 'quantizer.ckpt'

source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/finetune_cv_fr_6_1"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"

rsync -azh "$source_path" "$target_path" \
    --progress \
    --exclude 'optimizer.ckpt' \
    --exclude 'quantizer.ckpt' \
    --exclude 'results'