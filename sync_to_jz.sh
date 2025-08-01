source_path="/local_disk/apollon/rwhetten/sss_data_selection/training/results"
target_path="uaj64gk@jean-zay.idris.fr:/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/"

rsync -azh "$source_path" "$target_path" \
    --progress \
    --exclude 'optimizer.ckpt' \
    --exclude 'quantizer.ckpt'