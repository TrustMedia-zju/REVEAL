
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model llava-hf/llava-1.5-13b-hf \
    --train_type full \
    --dataset '/mnt/shangcephfs/mm-base-vision-ascend/sanmucao/datasets/benchmark/ft_local/updated_format_step1,2_think__mix_test_data.json' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 1 \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 4092 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir /mnt/shang2cephfs/mm-base-vision-ascend/sanmucao/pth/llava_pth/llava_13bhf1_1k \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --packing false \
    --use_liger_kernel true