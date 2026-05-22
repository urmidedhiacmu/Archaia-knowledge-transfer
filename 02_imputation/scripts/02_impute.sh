#!/bin/bash
#SBATCH --job-name=archaia_impute
#SBATCH --output=/home/udedhia/archaia_project/archaia_impute/logs/impute_%j.out
#SBATCH --error=/home/udedhia/archaia_project/archaia_impute/logs/impute_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

source /home/udedhia/archaia_env/bin/activate
source /home/udedhia/archaia_project/scripts/.env
export ARCHAIA_OPENAI_API_KEY
pip install faiss-cpu sentence-transformers torch torchvision openai requests pillow -q
cd /home/udedhia/archaia_project
python3 -u archaia_impute/02_impute/run.py
