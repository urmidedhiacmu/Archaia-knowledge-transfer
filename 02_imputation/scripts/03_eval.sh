#!/bin/bash
#SBATCH --job-name=archaia_eval
#SBATCH --output=/home/udedhia/archaia_project/archaia_impute/logs/eval_%j.out
#SBATCH --error=/home/udedhia/archaia_project/archaia_impute/logs/eval_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /home/udedhia/archaia_env/bin/activate
source /home/udedhia/archaia_project/scripts/.env
export ARCHAIA_OPENAI_API_KEY
pip install faiss-cpu sentence-transformers torch torchvision openai requests pillow transformers rapidfuzz nltk -q
cd /home/udedhia/archaia_project
python3 -u archaia_impute/03_eval/run_eval.py