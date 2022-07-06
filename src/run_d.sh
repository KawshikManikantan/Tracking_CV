#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -w gnode30
#SBATCH --mail-type=END,FAIL
#SBATCH --output=goturn_d.txt
source /home2/tgv2002/miniconda3/etc/profile.d/conda.sh
conda activate py37
cp -r ~/pygo /scratch
cd /scratch/pygo/src
papermill  --request-save-on-cell-execute --log-output --log-level INFO --progress-bar train1-withdrop.ipynb train1-withdrop_res.ipynb
