#!/bin/bash
#SBATCH --job-name=archaia_build_input
#SBATCH --output=/home/udedhia/archaia_project/archaia_impute/logs/build_input_%j.out
#SBATCH --error=/home/udedhia/archaia_project/archaia_impute/logs/build_input_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /home/udedhia/archaia_env/bin/activate
cd /home/udedhia/archaia_project
python3 -u archaia_impute/00_data_prep/build_input.py
