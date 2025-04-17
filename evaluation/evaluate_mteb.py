#!/usr/bin/env python
"""
MTEB Evaluation Script

This script evaluates sentence embedding models using the Massive Text Embedding
Benchmark (MTEB) framework, with a focus on Arabic language tasks.

Usage:
    python evaluate_mteb.py --model_name MODEL_NAME [--tasks TASK1,TASK2,...]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import mteb
import torch
from sentence_transformers import SentenceTransformer

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"mteb_evaluation_{os.path.basename(sys.argv[0])}.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate embedding models on MTEB benchmark")
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path to evaluate"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="STS17,STS22,STS22.v2",
        help="Comma-separated list of MTEB task names to evaluate"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="ara",
        help="Comma-separated list of languages to evaluate (e.g., 'ara,eng')"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for results (default: results/{model_name})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--truncate_dim",
        type=int,
        default=None,
        help="Truncate embeddings to this dimension (for Matryoshka models)"
    )
    
    return parser.parse_args()


def setup_device(device_arg: Optional[str] = None) -> str:
    """
    Set up the device for evaluation.
    
    Args:
        device_arg: Optional device specification

    Returns:
        Device string to use
    """
    if device_arg:
        device = device_arg
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # If specific GPUs are selected via environment variable
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if gpu_ids and gpu_ids[0]:  # If at least one GPU is specified
            device = f"cuda:{0}"  # Use the first available GPU
        else:
            device = "cpu"
    else:
        # Auto-detect if CUDA is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    return device


def get_model_wrapper(model_name: str, truncate_dim: Optional[int] = None, batch_size: int = 32, device: str = None):
    """
    Get a model wrapped for MTEB evaluation, with optional dimension truncation
    for Matryoshka models.
    
    Args:
        model_name: Model name or path
        truncate_dim: Dimension to truncate embeddings to (if provided)
        batch_size: Batch size for encoding
        device: Device to run model on
    
    Returns:
        Wrapped model ready for MTEB evaluation
    """
    device = setup_device(device)
    
    # Load the base model
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"Loaded model: {model_name}")
    
    # Check if we're using a custom dimension for Matryoshka models
    if truncate_dim is not None:
        logger.info(f"Using truncated dimension: {truncate_dim}")
        
        # Create a wrapper class that truncates embeddings
        class TruncatedModel:
            def __init__(self, base_model, dim):
                self.model = base_model
                self.truncate_dim = dim
                self.embedding_dimension = dim
            
            def encode(self, sentences, batch_size=batch_size, **kwargs):
                embeddings = self.model.encode(sentences, batch_size=batch_size, **kwargs)
                return embeddings[:, :self.truncate_dim]
        
        return TruncatedModel(model, truncate_dim)
    else:
        return model


def evaluate_mteb(model_name: str, tasks: List[str], languages: List[str], 
                 output_folder: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 32, truncate_dim: Optional[int] = None):
    """
    Evaluate a model on MTEB tasks.
    
    Args:
        model_name: Model name or path
        tasks: List of MTEB task names
        languages: List of languages to evaluate
        output_folder: Output folder for results
        device: Device to run model on
        batch_size: Batch size for encoding
        truncate_dim: Dimension to truncate embeddings to (for Matryoshka models)
    
    Returns:
        Dictionary with evaluation results
    """
    # Set up output folder
    if output_folder is None:
        model_short_name = model_name.split('/')[-1]
        dim_suffix = f"_dim{truncate_dim}" if truncate_dim else ""
        output_folder = f"results/{model_short_name}{dim_suffix}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the model
    model = get_model_wrapper(model_name, truncate_dim, batch_size, device)
    
    # Get MTEB tasks
    mteb_tasks = mteb.get_tasks(tasks=tasks, languages=languages)
    logger.info(f"Evaluating on {len(mteb_tasks)} tasks: {', '.join(tasks)}")
    
    # Initialize the MTEB evaluator
    evaluator = mteb.MTEB(tasks=mteb_tasks)
    
    # Run the evaluation
    results = evaluator.run(
        model, 
        output_folder=output_folder,
        batch_size=batch_size,
        verbosity=1  # Verbose output
    )
    
    # Save results to JSON file
    results_file = os.path.join(output_folder, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse task and language lists
    tasks = [task.strip() for task in args.tasks.split(",")]
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    # Run evaluation
    results = evaluate_mteb(
        model_name=args.model_name,
        tasks=tasks,
        languages=languages,
        output_folder=args.output_folder,
        device=args.device,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim
    )
    
    # Print summary of results
    logger.info("=== Evaluation Results Summary ===")
    for task_name, task_results in results.items():
        logger.info(f"Task: {task_name}")
        for metric, value in task_results.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        logger.info(f"  {metric}.{sub_metric}: {sub_value:.4f}")
            elif isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        logger.info("---")


if __name__ == "__main__":
    main()
