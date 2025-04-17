"""
Utility functions for model evaluation.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import util

logger = logging.getLogger(__name__)


def compute_similarity_metrics(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    gold_scores: List[float]
) -> Dict[str, float]:
    """
    Compute similarity metrics between embeddings and gold scores.
    
    Args:
        embeddings1: Embeddings of the first sentences
        embeddings2: Embeddings of the second sentences
        gold_scores: Gold similarity scores
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate cosine similarity scores
    cos_scores = []
    for i in range(len(embeddings1)):
        cos_scores.append(util.cos_sim(embeddings1[i], embeddings2[i]).item())
    
    # Convert everything to numpy for easier handling
    cos_scores = np.array(cos_scores)
    gold_scores = np.array(gold_scores)
    
    # Calculate Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(cos_scores, gold_scores)
    spearman_corr, _ = spearmanr(cos_scores, gold_scores)
    
    return {
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr),
        "cosine_similarity_avg": float(np.mean(cos_scores))
    }


def prepare_sts_data(task: str, split: str = "test") -> Tuple[List[str], List[str], List[float]]:
    """
    Prepare data for STS evaluation.
    
    Args:
        task: STS task name
        split: Dataset split (train, dev, test)
    
    Returns:
        Tuple of sentences1, sentences2, gold_scores
    """
    # Map task names to dataset names and configurations
    task_mapping = {
        "STS12": ("sentence-transformers/STS12", None),
        "STS13": ("sentence-transformers/STS13", None),
        "STS14": ("sentence-transformers/STS14", None),
        "STS15": ("sentence-transformers/STS15", None),
        "STS16": ("sentence-transformers/STS16", None),
        "STS17": ("sentence-transformers/STS17-ar", None),
        "STS17-ar": ("sentence-transformers/STS17-ar", None),
        "STS22": ("sentence-transformers/STS22-ar", None),
        "STS22-ar": ("sentence-transformers/STS22-ar", None),
        "STS22.v2": ("sentence-transformers/STS22-ar", "v2"),
        "STS22.v2-ar": ("sentence-transformers/STS22-ar", "v2"),
        "STSbenchmark": ("sentence-transformers/stsb", None),
        "stsb": ("sentence-transformers/stsb", None),
    }
    
    # Validate task name
    if task not in task_mapping:
        logger.warning(f"Unknown task: {task}. Available tasks: {', '.join(task_mapping.keys())}")
        return [], [], []
    
    dataset_name, config = task_mapping[task]
    
    try:
        # Handle special split names
        if split == "dev":
            split = "validation"
        
        # Load the dataset
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Extract sentence pairs and scores based on the dataset structure
        if "sentence1" in dataset.column_names and "sentence2" in dataset.column_names:
            sentences1 = dataset["sentence1"]
            sentences2 = dataset["sentence2"]
        elif "text_a" in dataset.column_names and "text_b" in dataset.column_names:
            sentences1 = dataset["text_a"]
            sentences2 = dataset["text_b"]
        else:
            raise ValueError(f"Could not identify sentence pair columns in dataset: {dataset.column_names}")
        
        # Extract scores
        if "score" in dataset.column_names:
            scores = dataset["score"]
        elif "similarity_score" in dataset.column_names:
            scores = dataset["similarity_score"]
        else:
            raise ValueError(f"Could not identify score column in dataset: {dataset.column_names}")
        
        # Normalize scores to [0, 1] range if necessary
        max_score = max(scores)
        if max_score > 1:
            # Assuming scores are on a 0-5 scale
            scores = [score / 5.0 for score in scores]
        
        logger.info(f"Loaded {len(sentences1)} sentence pairs from {task} ({split} split)")
        return sentences1, sentences2, scores
    
    except Exception as e:
        logger.error(f"Error loading {task} dataset: {str(e)}")
        return [], [], []


def evaluate_similarity(
    model,
    sentences1: List[str],
    sentences2: List[str],
    gold_scores: List[float],
    batch_size: int = 32,
    dimensions: Optional[List[int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate sentence similarity with a model.
    
    Args:
        model: Sentence embedding model
        sentences1: First sentences
        sentences2: Second sentences
        gold_scores: Gold similarity scores
        batch_size: Batch size for encoding
        dimensions: List of dimensions to evaluate (for Matryoshka models)
    
    Returns:
        Dictionary with evaluation results for each dimension
    """
    results = {}
    
    # Encode sentences
    embeddings1 = model.encode(sentences1, batch_size=batch_size, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size, convert_to_numpy=True)
    
    # Evaluate at full dimension
    full_metrics = compute_similarity_metrics(embeddings1, embeddings2, gold_scores)
    results["full_dim"] = full_metrics
    
    # Evaluate at specific dimensions if requested
    if dimensions:
        for dim in dimensions:
            if dim >= embeddings1.shape[1]:
                logger.warning(f"Requested dimension {dim} exceeds model dimension {embeddings1.shape[1]}")
                continue
            
            # Truncate embeddings to the specified dimension
            trunc_embeddings1 = embeddings1[:, :dim]
            trunc_embeddings2 = embeddings2[:, :dim]
            
            # Calculate metrics
            metrics = compute_similarity_metrics(trunc_embeddings1, trunc_embeddings2, gold_scores)
            results[f"dim_{dim}"] = metrics
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_file: str):
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results
        output_file: Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Determine file format based on extension
    extension = os.path.splitext(output_file)[1].lower()
    
    if extension == ".json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    elif extension == ".csv":
        # Flatten the results structure for CSV output
        flat_results = []
        for task_name, task_results in results.items():
            for dim_name, metrics in task_results.items():
                row = {"task": task_name, "dimension": dim_name}
                row.update(metrics)
                flat_results.append(row)
        
        # Write to CSV
        with open(output_file, "w", newline="") as f:
            if flat_results:
                writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
                writer.writeheader()
                writer.writerows(flat_results)
    else:
        # Default to JSON for unknown extensions
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def compare_models(
    results: Dict[str, Dict[str, Dict[str, float]]],
    baseline_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "spearman"
) -> Dict[str, Dict[str, float]]:
    """
    Compare evaluation results against a baseline.
    
    Args:
        results: Current model evaluation results
        baseline_results: Baseline model evaluation results
        metric: Metric to compare (e.g., "spearman", "pearson")
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for task, task_results in results.items():
        task_comparison = {}
        
        for dim, metrics in task_results.items():
            # Get current model metric
            current_value = metrics.get(metric, 0.0)
            
            # Get baseline metric if available
            baseline_value = 0.0
            if task in baseline_results and dim in baseline_results[task]:
                baseline_value = baseline_results[task][dim].get(metric, 0.0)
            
            # Calculate difference
            difference = current_value - baseline_value
            
            task_comparison[dim] = {
                "current": current_value,
                "baseline": baseline_value,
                "difference": difference,
                "relative_improvement": difference / baseline_value if baseline_value > 0 else float('inf')
            }
        
        comparison[task] = task_comparison
    
    return comparison
