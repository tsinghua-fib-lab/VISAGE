#!/bin/bash

# Set the master address to localhost for single node
export MASTER_ADDR="127.0.0.1"
export CURRENT_RANK=0
n_node=1

# Configurable parameters
DATASET_PREFIX=${1:-"segregation_cot_40_31_data"}
NUM_FOLDS=${2:-5}
START_FOLD=${3:-1}  # Optional: start from specific fold

# eg command: bash ./sequential_train.sh "my_dataset" 3 2

echo "MASTER_ADDR="$MASTER_ADDR
echo "Dataset prefix: $DATASET_PREFIX"
echo "Number of folds: $NUM_FOLDS"
echo "Starting from fold: $START_FOLD"
echo "Single node setup, no SLURM required."

bs=1
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $CURRENT_RANK
export CUDA_VISIBLE_DEVICES=0,2,3,5,7

# Base paths
BASE_OUTPUT_DIR="/data3/maruolong/VISAGE/data/cross_train/31"
INITIAL_MODEL="/data3/zhangxin/wuwen/vila-1.5/last_checkpoint/pretrain_att_35w_clean_textenhance"

# Generate dataset list
DATASETS=()
for i in $(seq $START_FOLD $NUM_FOLDS); do
    DATASETS+=("${DATASET_PREFIX}${i}")
done

echo "Datasets to train: ${DATASETS[@]}"

# Loop through datasets for sequential training
for i in "${!DATASETS[@]}"; do
    DATASET_NAME="${DATASETS[$i]}"
    ACTUAL_INDEX=$((START_FOLD + i - 1))
    
    echo "=========================================="
    echo "🚀 Training fold $((ACTUAL_INDEX + 1))/$NUM_FOLDS: $DATASET_NAME"
    echo "=========================================="
    
    # Determine input model path
    if [ $i -eq 0 ]; then
        MODEL_PATH=$INITIAL_MODEL
        echo "Using initial model: $MODEL_PATH"
    else
        PREV_DATASET="${DATASETS[$((i-1))]}"
        MODEL_PATH="$BASE_OUTPUT_DIR/${PREV_DATASET}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
        echo "Using previous model: $MODEL_PATH"
        
        # Check if previous model exists
        if [ ! -d "$MODEL_PATH" ]; then
            echo "❌ Previous model not found: $MODEL_PATH"
            echo "Please ensure previous training completed successfully"
            exit 1
        fi
    fi
    
    # Create output directory
    DATASET_OUTPUT_DIR="$BASE_OUTPUT_DIR/${DATASET_NAME}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
    mkdir -p $(dirname "$DATASET_OUTPUT_DIR")
    
    echo "Output directory: $DATASET_OUTPUT_DIR"
    
    # Run training
    torchrun --nnodes=$n_node --nproc_per_node=5 --master_port=25005 \
        --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
        /data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py \
        --deepspeed /data3/maruolong/UrbanMLLM/vila/scripts/zero3.json \
        --model_name_or_path $MODEL_PATH \
        --version llama_3 \
        --data_mixture $DATASET_NAME \
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
        --output_dir $DATASET_OUTPUT_DIR \
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
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo "❌ Training failed for $DATASET_NAME"
        exit 1
    fi
    
    echo "✅ Completed $DATASET_NAME"
    echo ""
done

echo "🎉 All $NUM_FOLDS folds training completed successfully!"