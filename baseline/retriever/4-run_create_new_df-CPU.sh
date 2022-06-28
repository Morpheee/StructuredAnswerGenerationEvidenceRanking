#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="create_new_df"

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=5

#SBATCH --output="./output/create_new_df.out"
#SBATCH --error="./output/create_new_df.err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                      create_new_df.py \
                      --test-new-path ./retrieved/dpr/sections_all_first_sentence_retrieves_sections_no_skipped_first_sentence/retrieved_w_heading_first_sentence.json \
                      --test-old-path ../../../data-set_pre_processed/test/sections_test.json \
                      --save-path ./retrieved/dpr/sections_all_first_sentence_retrieves_sections_no_skipped_first_sentence/df_test_retrieved.json

exit 0
EOT