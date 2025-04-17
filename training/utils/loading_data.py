"""
Utility functions for loading and processing datasets.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_training_data(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "train",
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> Dataset:
    """
    Load a dataset for training.
    
    Args:
        dataset_name: Name or path of the dataset
        dataset_config: Configuration name for the dataset (if applicable)
        split: Which split to load (train, validation, test)
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
    
    Returns:
        Loaded dataset
    """
    try:
        # Load the dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        
        # Select subset if specified
        if num_samples is not None and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            logger.info(f"Selected {num_samples} samples from dataset")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        raise


def load_multiple_datasets(
    dataset_configs: List[Dict],
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, Dataset]:
    """
    Load multiple datasets based on configuration.
    
    Args:
        dataset_configs: List of dictionaries with dataset configurations
        shuffle: Whether to shuffle the datasets
        seed: Random seed for shuffling
    
    Returns:
        Dictionary mapping dataset names to loaded datasets
    """
    datasets = {}
    
    for config in dataset_configs:
        name = config.get("name", config.get("dataset_name"))
        dataset_name = config.get("dataset_name")
        dataset_config = config.get("dataset_config")
        split = config.get("split", "train")
        num_samples = config.get("num_samples")
        
        if not name:
            name = f"{dataset_name}-{dataset_config}" if dataset_config else dataset_name
        
        try:
            dataset = load_training_data(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
                shuffle=shuffle,
                seed=seed
            )
            
            datasets[name] = dataset
            logger.info(f"Loaded dataset {name} with {len(dataset)} samples")
        
        except Exception as e:
            logger.warning(f"Failed to load dataset {name}: {str(e)}")
    
    return datasets


def get_sentence_pairs(dataset: Dataset, col1: str, col2: str) -> Tuple[List[str], List[str]]:
    """
    Extract sentence pairs from a dataset.
    
    Args:
        dataset: Dataset containing sentence pairs
        col1: Column name for the first sentence
        col2: Column name for the second sentence
    
    Returns:
        Tuple containing lists of first and second sentences
    """
    sentences1 = []
    sentences2 = []
    
    for example in dataset:
        if col1 in example and col2 in example:
            sentences1.append(str(example[col1]))
            sentences2.append(str(example[col2]))
    
    return sentences1, sentences2


def load_stsb_dataset(split: str = "train") -> Tuple[List[str], List[str], List[float]]:
    """
    Load the STS Benchmark dataset.
    
    Args:
        split: Dataset split to load (train, validation, test)
    
    Returns:
        Tuple containing lists of sentence1, sentence2, and scores
    """
    # Map split names to the names used in the dataset
    split_map = {
        "dev": "validation",
        "val": "validation",
        "validation": "validation",
        "train": "train",
        "test": "test"
    }
    
    mapped_split = split_map.get(split, split)
    
    try:
        dataset = load_dataset("sentence-transformers/stsb", split=mapped_split)
        
        sentences1 = dataset["sentence1"]
        sentences2 = dataset["sentence2"]
        scores = dataset["score"]
        
        # Normalize scores to [0, 1] if they are on a different scale
        if max(scores) > 1.0:
            scores = [score / 5.0 for score in scores]
        
        return sentences1, sentences2, scores
    
    except Exception as e:
        logger.error(f"Error loading STS Benchmark dataset: {str(e)}")
        return [], [], []


def load_nli_dataset(
    dataset_name: str = "sentence-transformers/all-nli",
    dataset_config: str = "triplet",
    num_samples: Optional[int] = None
) -> Union[Dataset, Dict[str, Dataset]]:
    """
    Load an NLI dataset for training.
    
    Args:
        dataset_name: Name of the dataset (default: sentence-transformers/all-nli)
        dataset_config: Configuration name (pair, triplet, pair-class, pair-score)
        num_samples: Number of samples to load (None for all)
    
    Returns:
        Loaded dataset or dictionary of datasets
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config)
        
        if isinstance(dataset, DatasetDict):
            # If split is not specified, return all splits
            if num_samples is not None:
                # Apply sample limit to each split
                for split in dataset:
                    if len(dataset[split]) > num_samples:
                        dataset[split] = dataset[split].select(range(num_samples))
                        logger.info(f"Limited {split} split to {num_samples} samples")
            
            return dataset
        else:
            # If dataset is already a single split
            if num_samples is not None and len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
                logger.info(f"Limited dataset to {num_samples} samples")
            
            return dataset
    
    except Exception as e:
        logger.error(f"Error loading NLI dataset {dataset_name}: {str(e)}")
        raise
