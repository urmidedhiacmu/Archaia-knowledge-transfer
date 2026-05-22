#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=upload_%j.log
#SBATCH --error=upload_%j.err

source ~/.bashrc

echo "Starting upload loop..."

while true; do
    echo "Running uploader at $(date)"
    python3 -u /home/udedhia/archaia_project/scripts/upload_to_hf.py

    EXIT_CODE=$?
    echo "Uploader exited with code $EXIT_CODE at $(date)"

    if grep -q "Upload complete" upload_${SLURM_JOB_ID}.log 2>/dev/null; then
        echo "Upload finished!"
        break
    fi

    echo "Restarting in 60 seconds..."
    sleep 60
done

echo "Job finished."