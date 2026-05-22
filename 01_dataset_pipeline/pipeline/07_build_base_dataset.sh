#!/bin/bash
#SBATCH --job-name=archaia_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=archaia_final_%j.out
#SBATCH --error=archaia_final_%j.err

echo "Starting finalize dataset job"
echo "Node: $(hostname)"
echo "Time: $(date)"

cd /home/udedhia/archaia_project/scripts

python3 finalize_dataset.py

echo "Finished"
echo "Time: $(date)"