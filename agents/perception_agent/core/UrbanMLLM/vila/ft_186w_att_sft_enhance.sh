
# Set the master address to localhost for single node
export MASTER_ADDR="127.0.0.1"
export CURRENT_RANK=0

# Since it's single node, we don't need worker_list or SLURM_JOB_NODELIST
n_node=1

echo "MASTER_ADDR="$MASTER_ADDR
echo "Single node setup, no SLURM required."


# for example, llava-v1.5-7b-mm-align


bs=1  # Adjust batch size as needed for your single GPU
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $CURRENT_RANK
export CUDA_VISIBLE_DEVICES=0,2,3,5,7

torchrun --nnodes=$n_node --nproc_per_node=5 --master_port=25005 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed /data3/maruolong/UrbanMLLM/vila/scripts/zero3.json \
    --model_name_or_path /data3/zhangxin/wuwen/vila-1.5/last_checkpoint/pretrain_att_35w_clean_textenhance \
    --version llama_3 \
    --data_mixture segregation_cot_40_31_data5 \
    --vision_tower /data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir /data3/maruolong/segregation/Train/UrbanMLLM/cross_train/31/data5/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new  \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 20000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to "tensorboard"
