#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="DPR"$1

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

#SBATCH --output="./output/dpr_retriever"$1".out"
#SBATCH --error="./output/dpr_retriever"$1".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 dpr_retriever.py --checkpoints-suffix $1
exit 0
EOT