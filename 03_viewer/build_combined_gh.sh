#!/bin/bash
#SBATCH --job-name=archaia_build
#SBATCH --output=/home/udedhia/archaia_project/scripts/build_combined_%j.out
#SBATCH --error=/home/udedhia/archaia_project/scripts/build_combined_%j.err
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /home/udedhia/archaia_env/bin/activate
pip install Pillow -q
cd /home/udedhia/archaia_project/scripts
python3 -u build_combined_gh.py