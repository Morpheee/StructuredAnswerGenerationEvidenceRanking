#!/bin/sh


#SBATCH --job-name=SHRID

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000

#SBATCH --output=shorten_id.out
#SBATCH --error=shorten_id.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 shorten_id.py


