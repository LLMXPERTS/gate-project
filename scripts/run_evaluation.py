#!/bin/bash
# Script to run evaluation for Matryoshka Embeddings

# Default parameters
MODEL_NAME=""
EVAL_TYPE="mteb"
CONFIG_FILE="configs/eval_config.json"
OUTPUT_DIR=""
TASKS=""
DIMENSIONS=""
DEVICE=""

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
            EVAL_TYPE="$2"
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
        --tasks)
            TASKS="$2"
            shift
            shift
            ;;
        --dimensions)
            DIMENSIONS="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model name is provided
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name is required. Use --model MODEL_NAME"
    exit 1
fi

# Set up output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_SHORT_NAME=$(basename "$MODEL_NAME")
    OUTPUT_DIR="results/$MODEL_SHORT_NAME"
fi

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Set up command based on evaluation type
if [ "$EVAL_TYPE" = "mteb" ]; then
    COMMAND="python evaluation/evaluate_mteb.py --model_name $MODEL_NAME --output_folder $OUTPUT_DIR"
    
    # Add tasks if specified
    if [ ! -z "$TASKS" ]; then
        COMMAND="$COMMAND --tasks $TASKS"
    fi
    
    # Add dimensions if specified
    if [ ! -z "$DIMENSIONS" ]; then
        COMMAND="$COMMAND --truncate_dim $DIMENSIONS"
    fi
    
    # Add device if specified
    if [ ! -z "$DEVICE" ]; then
        COMMAND="$COMMAND --device $DEVICE"
    fi
elif [ "$EVAL_TYPE" = "sts" ]; then
    COMMAND="python evaluation/evaluate_sts.py --model_name $MODEL_NAME --output_dir $OUTPUT_DIR"
    
    # Add task if specified
    if [ ! -z "$TASKS" ]; then
        COMMAND="$COMMAND --task $TASKS"
    fi
    
    # Add dimensions if specified
    if [ ! -z "$DIMENSIONS" ]; then
        COMMAND="$COMMAND --dimensions $DIMENSIONS"
    fi
    
    # Add device if specified
    if [ ! -z "$DEVICE" ]; then
        COMMAND="$COMMAND --device $DEVICE"
    fi
else
    echo "Unknown evaluation type: $EVAL_TYPE"
    echo "Available options: mteb, sts"
    exit 1
fi

# Log the command
echo "Running command: $COMMAND"
echo "Evaluation started at $(date)"

# Run the command
$COMMAND

# Log completion
echo "Evaluation completed at $(date)"
