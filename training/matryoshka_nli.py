#!/usr/bin/env python
"""
Training script for Matryoshka Embeddings on NLI dataset.

The system trains a transformer model (BERT, RoBERTa, DistilBERT, etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and contradictions are added as hard negatives.

The model is evaluated on the STS benchmark dataset at different output dimensions during training.

Usage:
    python matryoshka_nli.py [pretrained_transformer_model_name]
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)
from sentence_transformers.training_args import BatchSamplers

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project utilities
from training.utils.model_utils import setup_model
from training.utils.data_loading import load_training_data

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"matryoshka_nli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Matryoshka Embedding model on NLI data")
    
    parser.add_argument(
        "--model_name", 
        type=str,
        default="distilroberta-base",
        help="Pretrained transformer model name or path"
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="configs/training_config.json",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="Output directory for saving model checkpoints"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int,
        default=100,
        help="Evaluate model every X training steps"
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true",
        help="Push the trained model to the Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id", 
        type=str,
        default=None,
        help="Model ID for uploading to Hugging Face Hub"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load training configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default parameters.")
        return {}
    except json.JSONDecodeError:
        logger.warning(f"Error parsing config file {config_path}. Using default parameters.")
        return {}


def train_matryoshka_model(args, config):
    """Train Matryoshka model with the specified arguments and configuration."""
    # Determine model name and output directory
    model_name = args.model_name
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"output/matryoshka_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Get training parameters from config or arguments
    batch_size = config.get("batch_size", args.batch_size)
    num_train_epochs = config.get("num_epochs", args.epochs)
    max_seq_length = config.get("max_seq_length", args.max_seq_length) 
    matryoshka_dims = config.get("matryoshka_dims", [768, 512, 256, 128, 64])
    
    logger.info(f"Training Matryoshka model with dimensions: {matryoshka_dims}")
    logger.info(f"Model: {model_name}, Batch size: {batch_size}, Epochs: {num_train_epochs}")
    
    # 1. Load or create SentenceTransformer model
    model = setup_model(model_name, max_seq_length=max_seq_length)
    logger.info(f"Model loaded: {model}")
    
    # 2. Load the AllNLI dataset
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    logger.info(f"Training dataset loaded with {len(train_dataset)} samples")
    logger.info(f"Evaluation dataset loaded with {len(eval_dataset)} samples")
    
    # Optional: Limit training data for debugging
    if config.get("debug_mode", False):
        train_limit = config.get("debug_samples", 5000)
        train_dataset = train_dataset.select(range(train_limit))
        logger.info(f"Debug mode: limited training to {train_limit} samples")
    
    # 3. Define training loss
    inner_train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)
    
    # 4. Define evaluators for different dimensions
    stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    evaluators = []
    
    for dim in matryoshka_dims:
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=stsb_eval_dataset["sentence1"],
                sentences2=stsb_eval_dataset["sentence2"],
                scores=stsb_eval_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-dev-{dim}",
                truncate_dim=dim,
            )
        )
    
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])
    
    # 5. Define the training arguments
    training_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Training parameters:
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=config.get("warmup_ratio", 0.1),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=config.get("fp16", True),
        bf16=config.get("bf16", False),
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Better for MNRL
        seed=args.seed,
        # Tracking parameters:
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=args.eval_steps // 2,
        run_name=f"matryoshka-nli-{model_name.split('/')[-1]}",
    )
    
    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    
    trainer.train()
    
    # 7. Evaluate on the STS Benchmark test dataset
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    evaluators = []
    
    for dim in matryoshka_dims:
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                scores=test_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-test-{dim}",
                truncate_dim=dim,
            )
        )
    
    test_evaluator = SequentialEvaluator(evaluators)
    test_results = test_evaluator(model)
    
    # 8. Save the final model
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)
    logger.info(f"Model saved to {final_output_dir}")
    
    # 9. (Optional) Push to the Hugging Face Hub
    if args.push_to_hub:
        try:
            hub_model_id = args.hub_model_id or f"{model_name.split('/')[-1]}-nli-matryoshka"
            model.push_to_hub(hub_model_id)
            logger.info(f"Model pushed to the Hugging Face Hub: {hub_model_id}")
        except Exception:
            logger.error(
                f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}\n"
                f"To upload it manually, run: `model = SentenceTransformer('{final_output_dir}')`"
                f" followed by `model.push_to_hub('{model_name.split('/')[-1]}-nli-matryoshka')`"
            )
    
    return model, test_results


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train the model
    model, results = train_matryoshka_model(args, config)
    
    # Print final results
    logger.info("=== Final Evaluation Results ===")
    for name, score in results.items():
        logger.info(f"{name}: {score:.4f}")
