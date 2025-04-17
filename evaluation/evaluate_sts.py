#!/usr/bin/env python
"""
STS Benchmark Evaluation Script

This script evaluates sentence embedding models on STS (Semantic Textual Similarity)
benchmark datasets, with support for evaluating at different embedding dimensions
for Matryoshka models.

Usage:
    python evaluate_sts.py --model_name MODEL_NAME [--task TASK_NAME]
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import utility functions
from evaluation.utils.evaluation_utils import prepare_sts_data, compute_similarity_metrics

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"sts_evaluation_{os.path.basename(sys.argv[0])}.log")
    ]
)
logger = logging.getLogger(__name__)


# Dictionary of available STS datasets
STS_DATASETS = {
    "STS12": "https://huggingface.co/datasets/sentence-transformers/STS12",
    "STS13": "https://huggingface.co/datasets/sentence-transformers/STS13",
    "STS14": "https://huggingface.co/datasets/sentence-transformers/STS14",
    "STS15": "https://huggingface.co/datasets/sentence-transformers/STS15",
    "STS16": "https://huggingface.co/datasets/sentence-transformers/STS16",
    "STS17-ar": "https://huggingface.co/datasets/sentence-transformers/STS17-ar",
    "STS22-ar": "https://huggingface.co/datasets/sentence-transformers/STS22-ar",
    "STSbenchmark": "https://huggingface.co/datasets/sentence-transformers/stsb"
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate embedding models on STS benchmarks")
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path to evaluate"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help=f"STS task to evaluate (one of: {', '.join(STS_DATASETS.keys())} or 'all')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (e.g., 'test', 'dev', 'train')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/{model_name})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Comma-separated list of dimensions to evaluate for Matryoshka models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g., 'cuda:0', 'cpu')"
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


def load_model(model_name: str, device: str = None) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    
    Args:
        model_name: Model name or path
        device: Device to run model on
    
    Returns:
        Loaded SentenceTransformer model
    """
    device = setup_device(device)
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"Loaded model: {model_name}")
    return model


def evaluate_sts_dataset(
    model: SentenceTransformer,
    task: str,
    split: str = "test",
    batch_size: int = 32,
    dimensions: List[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model on an STS dataset at multiple dimensions (for Matryoshka models).
    
    Args:
        model: SentenceTransformer model
        task: STS task name
        split: Dataset split to evaluate
        batch_size: Batch size for encoding
        dimensions: List of dimensions to evaluate at (for Matryoshka models)
    
    Returns:
        Dictionary with evaluation results for each dimension
    """
    # Load the dataset
    sentences1, sentences2, gold_scores = prepare_sts_data(task, split)
    
    if not sentences1 or len(sentences1) == 0:
        logger.warning(f"No samples found for task {task}, split {split}")
        return {}
    
    logger.info(f"Evaluating {task} ({split} split) with {len(sentences1)} sentence pairs")
    
    # Encode sentences
    embeddings1 = model.encode(sentences1, batch_size=batch_size, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size, convert_to_numpy=True)
    
    # If dimensions are specified, evaluate at each dimension
    results = {}
    if dimensions:
        for dim in dimensions:
            dim_key = f"dim_{dim}"
            # Truncate embeddings to the specified dimension
            trunc_embeddings1 = embeddings1[:, :dim]
            trunc_embeddings2 = embeddings2[:, :dim]
            
            # Calculate similarity scores and metrics
            metrics = compute_similarity_metrics(trunc_embeddings1, trunc_embeddings2, gold_scores)
            results[dim_key] = metrics
            
            logger.info(f"Results for {task} at dimension {dim}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
    else:
        # Evaluate with full dimensionality
        metrics = compute_similarity_metrics(embeddings1, embeddings2, gold_scores)
        results["full_dim"] = metrics
        
        logger.info(f"Results for {task} at full dimension:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return results


def evaluate_all_sts(
    model: SentenceTransformer,
    tasks: List[str],
    split: str = "test",
    batch_size: int = 32,
    dimensions: List[int] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate a model on multiple STS datasets.
    
    Args:
        model: SentenceTransformer model
        tasks: List of STS task names
        split: Dataset split to evaluate
        batch_size: Batch size for encoding
        dimensions: List of dimensions to evaluate at (for Matryoshka models)
    
    Returns:
        Dictionary with evaluation results for each task and dimension
    """
    all_results = {}
    
    for task in tasks:
        task_results = evaluate_sts_dataset(
            model=model,
            task=task,
            split=split,
            batch_size=batch_size,
            dimensions=dimensions
        )
        all_results[task] = task_results
    
    return all_results


def main():
    """Main function."""
    args = parse_arguments()
    
    # Determine which tasks to evaluate
    if args.task.lower() == "all":
        tasks = list(STS_DATASETS.keys())
    else:
        tasks = [args.task]
    
    # Parse dimensions if provided
    dimensions = None
    if args.dimensions:
        dimensions = [int(dim.strip()) for dim in args.dimensions.split(",")]
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_short_name = args.model_name.split("/")[-1]
        output_dir = f"results/{model_short_name}/sts"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_name, args.device)
    
    # Run evaluation
    results = evaluate_all_sts(
        model=model,
        tasks=tasks,
        split=args.split,
        batch_size=args.batch_size,
        dimensions=dimensions
    )
    
    # Save results
    output_file = os.path.join(output_dir, f"sts_results_{args.split}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("=== Summary of Results ===")
    for task, task_results in results.items():
        logger.info(f"Task: {task}")
        for dim, metrics in task_results.items():
            spearman = metrics.get("spearman", 0.0)
            logger.info(f"  {dim}: Spearman = {spearman:.4f}")
    
    # Calculate average across all tasks
    avg_results = {}
    for task, task_results in results.items():
        for dim, metrics in task_results.items():
            if dim not in avg_results:
                avg_results[dim] = {"count": 0, "spearman_sum": 0.0}
            
            avg_results[dim]["count"] += 1
            avg_results[dim]["spearman_sum"] += metrics.get("spearman", 0.0)
    
    logger.info("Average Spearman correlation across all tasks:")
    for dim, stats in avg_results.items():
        avg_spearman = stats["spearman_sum"] / stats["count"] if stats["count"] > 0 else 0.0
        logger.info(f"  {dim}: {avg_spearman:.4f}")


if __name__ == "__main__":
    main()
