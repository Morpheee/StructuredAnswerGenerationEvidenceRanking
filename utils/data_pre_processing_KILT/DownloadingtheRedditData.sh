#!/bin/sh


#SBATCH --job-name=DRDKILT

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=4

#SBATCH --output=DownloadingtheRedditData.out
#SBATCH --error=DownloadingtheRedditData.err

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 ./ELI5/data_creation/download_reddit_qalist.py -Q
srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 ./ELI5/data_creation/download_reddit_qalist.py -A