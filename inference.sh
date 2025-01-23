CUDA_VISIBLE_DEVICES=0 torchrun \
     --nnodes=1 \
     --nproc_per_node=1 \
     --master_port=21547 \
     inference.py \
     -conf ./config/test.yml