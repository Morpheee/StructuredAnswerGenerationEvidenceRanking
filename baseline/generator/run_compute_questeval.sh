#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="QuestEval"$1

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

#SBATCH --output="./output/QuestEval"$1".out"
#SBATCH --error="./output/QuestEval"$1".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      compute_questeval.py --df-output-path ./test_outputs/$1/test.json
exit 0
EOT