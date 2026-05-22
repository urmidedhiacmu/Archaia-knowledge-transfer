#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --output=ablations_%j.out
#SBATCH --error=ablations_%j.err
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source ~/archaia_env/bin/activate


cd /home/udedhia/archaia_project/scripts
python3 run_ablations_uses.py