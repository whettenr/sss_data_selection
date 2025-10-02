#!/bin/bash

# Usage: ./replace_paths.sh input_file output_file

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

input_file="$1"
output_file="$2"

# Create a temp file to build sed commands
tmp_sed=$(mktemp)

# Add mappings as sed substitution rules
cat > "$tmp_sed" <<'EOF'
s|/corpus/LeBenchmark/slr57_flowbert/African_Accented_French/wavs|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/African_Accented_French/wavs|g
s|/corpus/LeBenchmark/slr88_flowbert/Att-HACK_SLR88|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88|g
s|/corpus/LeBenchmark/cafe_flowbert/CaFE|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/CaFE|g
s|/corpus/LeBenchmark/CFPP_corrected/CFPP_corrected/output|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/CFPP_corrected/output|g
s|/corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves|g
s|/corpus/LeBenchmark/gemep_flowbert/GEMEP|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/GEMEP|g
s|/corpus/LeBenchmark/MaSS/MaSS/output_waves|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Mass/output_waves|g
s|/corpus/LeBenchmark/mpf_flowbert/MPF/output_waves|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MPF/output_waves|g
s|/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MultilingualLibriSpeech/mls_french|g
s|/corpus/LeBenchmark/nccfr_flowbert/NCCFr/output_waves|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/NCCFr/output_waves|g
s|/corpus/LeBenchmark/port_media_flowbert/PMDOM2FR_00/PMDOM2FR_wavs|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Portmedia/PMDOM2FR_wavs|g
s|/corpus/LeBenchmark/TCOF_corrected/TCOF_corrected/output|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/TCOF_corrected/output|g
s|/corpus/LeBenchmark/voxpopuli_transcribed_data/wav|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_transcribed/wav|g
s|/corpus/LeBenchmark/voxpopuli_unlabelled/wav|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav|g
s|/corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns|/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns|g
EOF

# %s/output_waves\/output_waves/output_waves/g
# Apply sed rules
sed -f "$tmp_sed" "$input_file" > "$output_file"

# Clean up
rm "$tmp_sed"

echo "Replacements complete. Output saved to $output_file"

# /lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MultilingualLibriSpeech/mls_french/train/audio/12823/13528/12823_13528_000195.wav

# /corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french/train/audio/12823/13528/12823_13528_000195.wav