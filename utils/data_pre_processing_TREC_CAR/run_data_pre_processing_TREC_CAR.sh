#!/bin/sh


#SBATCH --job-name=PPTC

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000

#SBATCH --output=data_pre_processing_TREC_CAR.out
#SBATCH --error=data_pre_processing_TREC_CAR.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 data_pre_processing_TREC_CAR.py


