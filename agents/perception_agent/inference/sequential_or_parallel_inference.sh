#!/bin/bash

# Flexible Inference Script with mode selection and customizable GPU allocation

# Configuration
BASE_MODEL_DIR="/data3/maruolong/VISAGE/data/cross_train/31"
BASE_DATA_DIR="/data3/maruolong/VISAGE/data/31_cities"
SCRIPT_PATH="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py"
CONV_MODE="llama_3"
TEMPERATURE=0.2

# Parse arguments
MODE=${1:-"sequential"}  # "sequential" or "parallel"
SPECIFIC_DATASET=${2:-""}  # Optional: run only specific dataset
GPU_LIST=${3:-"0,1,2,3,4,5,6,7"}  # Optional: specify GPU IDs

# Convert GPU list string to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"

# Detect available datasets
detect_datasets() {
    local base_dir="$1"
    local datasets=()
    
    for dir in "$base_dir"/*; do
        if [ -d "$dir" ]; then
            local dataset_name=$(basename "$dir")
            if [[ $dataset_name =~ data[0-9]+ ]]; then
                datasets+=("$dataset_name")
            fi
        fi
    done
    
    printf '%s\n' "${datasets[@]}" | sort -V
}

# Get datasets
if [ -n "$SPECIFIC_DATASET" ]; then
    DATASETS=("$SPECIFIC_DATASET")
    echo "🎯 Running only for specific dataset: $SPECIFIC_DATASET"
else
    DATASETS=($(detect_datasets "$BASE_MODEL_DIR"))
fi

NUM_FOLDS=${#DATASETS[@]}

if [ $NUM_FOLDS -eq 0 ]; then
    echo "❌ No datasets found in $BASE_MODEL_DIR"
    exit 1
fi

echo "📁 Found $NUM_FOLDS datasets: ${DATASETS[@]}"
echo "🔧 Mode: $MODE"
echo "🎮 Using GPUs: ${GPUS[@]} (Total: ${#GPUS[@]} GPUs)"

# Function to run inference
run_inference() {
    local dataset="$1"
    local data_type="$2"
    local gpu_id="$3"
    
    local model_path="$BASE_MODEL_DIR/$dataset/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
    local question_file="$BASE_DATA_DIR/$dataset/${data_type}_data.json"
    local output_file="$BASE_DATA_DIR/$dataset/feature_vector_segregation_${data_type}.pkl"
    
    echo "🔍 Running ${data_type} inference for $dataset on GPU $gpu_id"
    
    # Validation checks
    if [ ! -d "$model_path" ]; then
        echo "❌ Model not found: $model_path"
        return 1
    fi
    
    if [ ! -f "$question_file" ]; then
        echo "❌ Question file not found: $question_file"
        return 1
    fi
    
    mkdir -p "$(dirname "$output_file")"
    
    # Run inference
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --model-path "$model_path" \
        --question-file "$question_file" \
        --outputs-file "$output_file" \
        --image-folder "" \
        --temperature "$TEMPERATURE" \
        --conv-mode "$CONV_MODE"
    
    return $?
}

# Sequential execution
run_sequential() {
    echo "🔄 Starting sequential inference..."
    
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu_id="${GPUS[$((i % ${#GPUS[@]}))]}"
        
        echo ""
        echo "📊 Processing dataset: $dataset"
        echo "================================"
        
        # Train inference
        if run_inference "$dataset" "train" "$gpu_id"; then
            echo "✅ Train inference completed for $dataset"
        else
            echo "❌ Train inference failed for $dataset"
        fi
        
        # Test inference
        if run_inference "$dataset" "test" "$gpu_id"; then
            echo "✅ Test inference completed for $dataset"
        else
            echo "❌ Test inference failed for $dataset"
        fi
        
        echo "✅ Completed all inference for $dataset"
    done
}

# Parallel execution
run_parallel() {
    echo "🔄 Starting parallel inference..."
    declare -a PIDS=()
    
    # Calculate how many datasets per GPU
    local datasets_per_gpu=$(( (NUM_FOLDS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
    echo "📊 Distribution: Up to $datasets_per_gpu datasets per GPU"
    
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        # Assign GPU in round-robin fashion
        gpu_index=$((i % ${#GPUS[@]}))
        gpu_id="${GPUS[$gpu_index]}"
        
        (
            echo "🚀 Starting parallel job for $dataset on GPU $gpu_id"
            
            # Train inference
            if run_inference "$dataset" "train" "$gpu_id"; then
                echo "✅ Train inference completed for $dataset on GPU $gpu_id"
            else
                echo "❌ Train inference failed for $dataset on GPU $gpu_id"
            fi
            
            # Test inference
            if run_inference "$dataset" "test" "$gpu_id"; then
                echo "✅ Test inference completed for $dataset on GPU $gpu_id"
            else
                echo "❌ Test inference failed for $dataset on GPU $gpu_id"
            fi
            
            echo "🎉 Completed parallel inference for $dataset on GPU $gpu_id"
        ) &
        
        PIDS+=($!)
        echo "📊 Launched inference for $dataset on GPU $gpu_id (PID: ${PIDS[-1]})"
    done
    
    # Wait for completion
    echo ""
    echo "⏳ Waiting for all parallel jobs..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
}

# Show usage information
show_usage() {
    echo "Usage: $0 [mode] [dataset] [gpu_list]"
    echo ""
    echo "Arguments:"
    echo "  mode: 'sequential' or 'parallel' (default: sequential)"
    echo "  dataset: Specific dataset name (default: all datasets)"
    echo "  gpu_list: Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)"
    echo ""
    echo "Examples:"
    echo "  $0 sequential                          # Sequential with all GPUs"
    echo "  $0 parallel                            # Parallel with all GPUs"  
    echo "  $0 parallel \"\" \"0,1,2\"               # Parallel with GPUs 0,1,2"
    echo "  $0 sequential segregation_cot_40_31_data2 \"3\"  # Specific dataset on GPU 3"
    echo "  $0 parallel \"\" \"4,5,6,7\"             # Parallel with GPUs 4,5,6,7"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Main execution
case $MODE in
    "sequential")
        run_sequential
        ;;
    "parallel")
        run_parallel
        ;;
    *)
        echo "❌ Unknown mode: $MODE. Use 'sequential' or 'parallel'"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo "🎉 All inference completed!"


: '
================================================================================
USAGE EXAMPLES AND DOCUMENTATION
================================================================================

This script performs inference on K-fold cross-validation datasets with flexible
GPU allocation and execution modes.

BASIC USAGE:
  ./inference_flexible.sh [mode] [dataset] [gpu_list]

PARAMETERS:
  mode: Execution mode
    - "sequential": Run datasets one after another (default)
    - "parallel":   Run all datasets simultaneously
  
  dataset: Specific dataset to run (optional)
    - If empty: run all detected datasets
    - Example: "segregation_cot_40_31_data2"
  
  gpu_list: Comma-separated GPU IDs (optional)
    - Default: "0,1,2,3,4,5,6,7"
    - Example: "0,1,2" for GPUs 0, 1, and 2

EXECUTION MODES:

1. Sequential Mode:
   - Processes one dataset at a time
   - Each dataset uses one GPU (round-robin assignment)
   - Suitable when GPU memory is limited
   - Example: ./inference_flexible.sh sequential "" "0,1"

2. Parallel Mode:
   - Runs all datasets simultaneously
   - Distributes datasets across available GPUs
   - Each GPU can handle multiple datasets
   - Maximum parallelism for faster execution
   - Example: ./inference_flexible.sh parallel "" "0,1,2,3"

COMMON USAGE SCENARIOS:

# 1. Basic sequential inference with all GPUs
./inference_flexible.sh sequential

# 2. Parallel inference with all GPUs
./inference_flexible.sh parallel

# 3. Parallel inference with specific GPUs only
./inference_flexible.sh parallel "" "0,1,2"

# 4. Run only one specific dataset on a single GPU
./inference_flexible.sh sequential segregation_cot_40_31_data2 "3"

# 5. Parallel inference using only high-numbered GPUs
./inference_flexible.sh parallel "" "4,5,6,7"

# 6. Single GPU sequential inference
./inference_flexible.sh sequential "" "0"

# 7. Get help information
./inference_flexible.sh -h

OUTPUT:
  For each dataset, the script generates:
  - feature_vector_segregation_train.pkl (train set embeddings)
  - feature_vector_segregation_test.pkl  (test set embeddings)
  
  Files are saved in: /data3/maruolong/VISAGE/data/31_cities/{dataset_name}/

NOTES:
  - The script automatically detects available datasets by scanning the model directory
  - Each dataset runs both train and test inference sequentially
  - In parallel mode, multiple datasets may share the same GPU
  - Make sure the model checkpoints exist before running inference
  - Use nvidia-smi to check GPU availability and memory usage
================================================================================
'