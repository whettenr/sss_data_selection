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

