#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="BARToutline"$1

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

#SBATCH --output="./output/BART_generator_outline"$1".out"
#SBATCH --error="./output/BART_generator_outline"$1".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      BART_generator_outline.py --checkpoints-suffix "outline"$1
                      
exit 0
EOT