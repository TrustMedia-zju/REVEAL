# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model llava-hf/llava-1.5-7b-hf \
    --model_type llava1_5_hf \
    --train_type full \
    --dataset /mnt/shangcephfs/mm-base-vision-ascend/sanmucao/datasets/benchmark/ft_local/swift1_think_data.json\
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --packing false \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-32B \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
