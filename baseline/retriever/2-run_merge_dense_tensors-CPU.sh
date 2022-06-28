#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="merge"

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=5

#SBATCH --output="./output/merge.out"
#SBATCH --error="./output/merge.err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      merge_dense_tensors.py --directory ./tensors/dpr/sections_first_sentence
exit 0
EOT