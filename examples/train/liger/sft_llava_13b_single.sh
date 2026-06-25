# test env: 4 * A100
# Using use_liger_kernel and packing: 4 * 42GB, 1 hour 35 minutes
# Not using use_liger_kernel: 4 * 54GB, 1 hour 40 minutes
# Not using use_liger_kernel and packing: 4 * 52GB, 3 hours 30 minutes

# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model llava-hf/llava-1.5-13b-hf \
    --model_type llava1_5_hf \
    --train_type full \
    --dataset /mnt/shangcephfs/mm-base-vision-ascend/sanmucao/datasets/benchmark/REVEAL/REVEAL-Bench.json\
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --packing false \
    --num_train_epochs 10 \
    --save_steps 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir /mnt/shangcephfs/mm-base-vision-ascend/sanmucao/code/ms-swift/examples/train/liger/llavahf13b \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
