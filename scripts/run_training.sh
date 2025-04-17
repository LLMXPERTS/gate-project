#!/bin/bash
# Script to run training for Matryoshka Embeddings

# Check if CUDA is available and set visible devices
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Auto-detect GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ $NUM_GPUS -gt 0 ]; then
        # Use all available GPUs
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        echo "Auto-detected GPUs: $CUDA_VISIBLE_DEVICES"
    else
        echo "No GPUs detected, using CPU"
        CUDA_VISIBLE_DEVICES=""
    fi
fi

# Default parameters
MODEL_NAME="bert-base-uncased"
TRAINING_TYPE="matryoshka_nli"
CONFIG_FILE="configs/training_config.json"
OUTPUT_DIR="output/$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=0  # 0 means use value from config file
EPOCHS=0  # 0 means use value from config file
PUSH_TO_HUB=false
HUB_MODEL_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL_NAME="$2"
            shift
            shift
            ;;
        --type)
            TRAINING_TYPE="$2"
            shift
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --push-to-hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --hub-model-id)
            HUB_MODEL_ID="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Set up command based on training type
if [ "$TRAINING_TYPE" = "matryoshka_nli" ]; then
    COMMAND="python training/matryoshka_nli.py --model_name $MODEL_NAME --config $CONFIG_FILE --output_dir $OUTPUT_DIR"
    
    # Add optional parameters
    if [ $BATCH_SIZE -gt 0 ]; then
        COMMAND="$COMMAND --batch_size $BATCH_SIZE"
    fi
    
    if [ $EPOCHS -gt 0 ]; then
        COMMAND="$COMMAND --epochs $EPOCHS"
    fi
    
    if [ "$PUSH_TO_HUB" = true ]; then
        COMMAND="$COMMAND --push_to_hub"
        
        if [ ! -z "$HUB_MODEL_ID" ]; then
            COMMAND="$COMMAND --hub_model_id $HUB_MODEL_ID"
        fi
    fi
elif [ "$TRAINING_TYPE" = "hybrid" ]; then
    COMMAND="python training/hybrid_training.py --model_name $MODEL_NAME --config $CONFIG_FILE --output_dir $OUTPUT_DIR"
    
    # Add optional parameters
    if [ $BATCH_SIZE -gt 0 ]; then
        COMMAND="$COMMAND --batch_size $BATCH_SIZE"
    fi
    
    if [ $EPOCHS -gt 0 ]; then
        COMMAND="$COMMAND --epochs $EPOCHS"
    fi
    
    if [ "$PUSH_TO_HUB" = true ]; then
        COMMAND="$COMMAND --push_to_hub"
        
        if [ ! -z "$HUB_MODEL_ID" ]; then
            COMMAND="$COMMAND --hub_model_id $HUB_MODEL_ID"
        fi
    fi
else
    echo "Unknown training type: $TRAINING_TYPE"
    echo "Available options: matryoshka_nli, hybrid"
    exit 1
fi

# Log the command
echo "Running command: $COMMAND"
echo "Training started at $(date)"

# Run the command
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $COMMAND

# Log completion
echo "Training completed at $(date)"
