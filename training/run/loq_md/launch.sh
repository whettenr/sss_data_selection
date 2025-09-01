#!/bin/bash

# Loop over all .sh files in the current directory
for script in *.sh; do
    # Submit the job and capture the job ID
    job_submit_output=$(sbatch "$script")
    # sbatch output is usually: "Submitted batch job 123456"
    job_id=$(echo "$job_submit_output" | awk '{print $4}')

    echo "Submitted $script as job $job_id"

    # Submit the same job with dependency on the first
    sbatch --dependency=afterany:$job_id "$script"
    echo "Submitted $script again with dependency on job $job_id"
done