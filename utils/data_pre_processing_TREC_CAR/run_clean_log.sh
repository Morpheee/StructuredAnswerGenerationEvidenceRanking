#!/bin/sh


#SBATCH --job-name=PPTC

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=1

#SBATCH --output=clean_log.out
#SBATCH --error=clean_log.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 clean_log.py
