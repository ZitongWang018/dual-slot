# Matched to hyq718/p2n dd08fc2b 70M recipe.
architecture=(--num-layers 6 --hidden-size 512 --ffn-hidden-size 2048
              --num-attention-heads 8 --num-query-groups 2
              --untie-embeddings-and-output-weights)
schedule=(--micro-batch-size 4 --global-batch-size 256
          --train-iters 2836 --lr-warmup-iters 142)
