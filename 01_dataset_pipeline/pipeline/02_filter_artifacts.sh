#!/bin/bash
#SBATCH --job-name=archaia_filter
#SBATCH --output=/home/udedhia/archaia_project/scripts/filter_artifacts_%j.out
#SBATCH --error=/home/udedhia/archaia_project/scripts/filter_artifacts_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

source /home/udedhia/archaia_env/bin/activate
cd /home/udedhia/archaia_project/scripts
python3 -u filter_artifacts.py