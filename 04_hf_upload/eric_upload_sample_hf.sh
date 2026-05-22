#!/bin/bash
#SBATCH --job-name=archaia_sample_hf
#SBATCH --output=/home/udedhia/archaia_project/scripts/sample_upload_%j.out
#SBATCH --error=/home/udedhia/archaia_project/scripts/sample_upload_%j.err
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

source /home/udedhia/archaia_env/bin/activate
cd /home/udedhia/archaia_project/scripts
python3 -u create_sample_upload_hf_eric.py