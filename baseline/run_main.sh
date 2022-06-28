#!/bin/sh


#SBATCH --job-name=SAGER
#SBATCH --cpus-per-task=1

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1

#SBATCH --output=main.out
#SBATCH --error=main.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 main.py