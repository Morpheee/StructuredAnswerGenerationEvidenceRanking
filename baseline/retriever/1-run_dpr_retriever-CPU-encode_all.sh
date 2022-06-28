#!/bin/bash

splits=8

for (( i =1; i<= $splits; i++ ))
do
#echo dpr_retriever.py --checkpoints-suffix enc$i --encode-all-nb-splits $splits --encode-all-index-split $i --path-input-ids-tensor ./tensors/dpr/input_ids_tensor.pt --path-attention-masks-tensor ./tensors/dpr/attention_mask_tensor.pt --load-from-checkpoint ./checkpoints/epoch=14-step=16140.ckpt

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="DPRenc"$i

#SBATCH --partition=24CPUNodes
#SBATCH --cpus-per-task=5

#SBATCH --output="./output/dpr_retrieverENC"$1$i"_"$splits".out"
#SBATCH --error="./output/dpr_retrieverENC"$1$i"_"$splits".err"

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif /users/iris/rserrano/CAG-env-t5/bin/python3 \
                    dpr_retriever.py \
                    --checkpoints-suffix enc$i \
                    --suffix sections_first_sentence \
                    --encode-all-nb-splits $splits \
                    --encode-all-index-split $i \
                    --load-from-checkpoint ./checkpoints/sections_first_sentence/dpr_retriever/version_1/checkpoints/epoch=5-step=56586.ckpt \
                    --text-column "w_heading_first_sentence" \
                    --test-ds test/sections_test.json

exit 0
EOT

done
                    # --path-input-ids-tensor ./tensors/dpr/input_ids_tensor.pt \
                    # --path-attention-masks-tensor ./tensors/dpr/attention_mask_tensor.pt 