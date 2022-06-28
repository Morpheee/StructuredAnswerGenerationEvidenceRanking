#!/bin/sh


#SBATCH --job-name=SHRID

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=4

#SBATCH --output=shorten_id1.out
#SBATCH --error=shorten_id1.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 shorten_id.py


