#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="DPRtest"$1

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

#SBATCH --output="./output/dpr_retriever_test"$1".out"
#SBATCH --error="./output/dpr_retriever_test"$1".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      dpr_retriever.py \
                      --checkpoints-suffix test$1 \
                      --path-corpus-dense-tensor "./tensors/dpr/sections_first_sentence/dense.ckpt" \
                      --load-from-checkpoint ./checkpoints/sections_first_sentence/dpr_retriever/version_1/checkpoints/epoch=5-step=56586.ckpt \
                      --suffix "./tensors/dpr/sections_all_first_sentence_retrieves_sections_no_skipped_first_sentence/" \
                      --text-column "w_heading_first_sentence" \
                      --test-ds test/sections_test.json \

exit 0
EOT
                      # --test-ds-skipped test/skipped_sections_test.json