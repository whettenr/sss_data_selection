source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/results"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"

rsync -azh "$source_path" "$target_path" \
    --progress \
    --exclude 'optimizer.ckpt' \
    --exclude 'quantizer.ckpt' \
    --exclude 'save'


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