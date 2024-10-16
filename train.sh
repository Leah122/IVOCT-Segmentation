#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --container-mounts=/data/diag:/data/diag\ \
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=high
#SBATCH --output=/data/diag/leahheil/IVOCT-Segmentation/slurm-output/%j.out

python3 /data/diag/leahheil/IVOCT-Segmentation/train.py

