WANDB_NAME="grpo_thinkllava13b_stepthink_answer_vl" \
WANDB_API_KEY=c8ee6e546d7f62acdb1a30056dd1e7105b56968a \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model llava-hf/llava-1.5-13b-hf \
    --model_type llava1_5_hf \
    --external_plugins /mnt/shangcephfs/mm-base-vision-ascend/sanmucao/code/ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/shangcephfs/mm-base-vision-ascend/sanmucao/datasets/benchmark/REVEAL/REVEAL-Bench-GRPO.json \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --max_pixels 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /mnt/shang2cephfs/mm-base-vision-ascend/sanmucao/pth/llava_pth/GRPO_STEPTHINKllava13b \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 