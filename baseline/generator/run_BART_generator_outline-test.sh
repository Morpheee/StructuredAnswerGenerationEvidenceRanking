#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="BARTtest"$1

#SBATCH --partition=GPUNodes
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

#SBATCH --output="./output/BART_generator_outline_test"$1".out"
#SBATCH --error="./output/BART_generator_outline_test"$1".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      BART_generator_outline.py \
                      --checkpoints-suffix test_$1 \
                      --nb-passages 10 \
                      --suffix ./test_outputs/$1 \
                      --load-from-checkpoint ./checkpoints/outline/BART_generator/version_0/checkpoints/epoch=3-step=51188.ckpt \
                      --text-column w_heading_all_passage \
                      --test-corpus test/corpus_test.json \
                      --test-ds test/sections_test.json \

                    
exit 0
EOT
                     # --test-ds "../retriever/retrieved/dpr/sections_first_sentence_no_skipped/df_test_retrieved.json" \
                     # --test-prefix False