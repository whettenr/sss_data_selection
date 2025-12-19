#!/bin/bash

# Usage: ./replace_mls.sh input_file output_file

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

input_file="$1"
output_file="$2"

# Only replace .wav with .flac on lines containing "MultilingualLibriSpeech"
sed '/MultilingualLibriSpeech/ s/\.wav/\.flac/g' "$input_file" > "$output_file"

echo "Replacement complete. Output saved to $output_file"

sh process_csv.sh /local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/lebench_lg_40sec/length_0.5.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg_40/jz_length_0.5.csv