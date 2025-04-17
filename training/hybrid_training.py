#!/usr/bin/env python
"""
Hybrid multi-dataset training script.

This script implements a hybrid training approach that combines multiple datasets
and loss functions to create robust sentence embeddings. It supports training with
different datasets simultaneously, applying appropriate loss functions for each.

Usage:
    python hybrid_training.py [--model_name MODEL_NAME] [--config CONFIG_FILE]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
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

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project utilities
from training.utils.model_utils import setup_model

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"hybrid_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with hybrid multi-dataset approach")
    
    parser.add_argument(
        "--model_name", 
        type=str,
        default="bert-base-uncased",
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
        default=64,
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
        default=1000,
        help="Evaluate model every X training steps"
    )
    parser.add_argument(
        "--samples_per_dataset", 
        type=int,
        default=10000,
        help="Number of samples to use from each dataset"
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
    parser.add_argument(
        "--use_matryoshka", 
        action="store_true",
        help="Use MatryoshkaLoss wrapper around losses"
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


def train_hybrid_model(args, config):
    """Train model with hybrid multi-dataset approach."""
    # Determine model name and output directory
    model_name = args.model_name
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_short_name = model_name.split('/')[-1]
        output_dir = f"output/hybrid_{model_short_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Get training parameters from config or arguments
    batch_size = config.get("batch_size", args.batch_size)
    num_train_epochs = config.get("num_epochs", args.epochs)
    max_seq_length = config.get("max_seq_length", args.max_seq_length)
    samples_per_dataset = config.get("samples_per_dataset", args.samples_per_dataset)
    matryoshka_dims = config.get("matryoshka_dims", [768, 512, 256, 128, 64])
    use_matryoshka = config.get("use_matryoshka", args.use_matryoshka)
    
    logger.info(f"Training hybrid model: {model_name}")
    logger.info(f"Batch size: {batch_size}, Epochs: {num_train_epochs}")
    if use_matryoshka:
        logger.info(f"Using Matryoshka loss with dimensions: {matryoshka_dims}")
    
    # 1. Load or create SentenceTransformer model
    model = setup_model(model_name, max_seq_length=max_seq_length)
    logger.info(f"Model loaded: {model}")
    
    # 2. Load multiple datasets for training
    
    # (anchor, positive)
    all_nli_pair_train = load_dataset(
        "sentence-transformers/all-nli", "pair", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (premise, hypothesis) + label
    all_nli_pair_class_train = load_dataset(
        "sentence-transformers/all-nli", "pair-class", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (sentence1, sentence2) + score
    all_nli_pair_score_train = load_dataset(
        "sentence-transformers/all-nli", "pair-score", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (anchor, positive, negative)
    all_nli_triplet_train = load_dataset(
        "sentence-transformers/all-nli", "triplet", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (sentence1, sentence2) + score
    stsb_pair_score_train = load_dataset(
        "sentence-transformers/stsb", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (anchor, positive)
    quora_pair_train = load_dataset(
        "sentence-transformers/quora-duplicates", "pair", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # (query, answer)
    natural_questions_train = load_dataset(
        "sentence-transformers/natural-questions", 
        split=f"train[:{samples_per_dataset}]"
    )
    
    # Combine all datasets into a dictionary with dataset names to datasets
    train_dataset = {
        "all-nli-pair": all_nli_pair_train,
        "all-nli-pair-class": all_nli_pair_class_train,
        "all-nli-pair-score": all_nli_pair_score_train,
        "all-nli-triplet": all_nli_triplet_train,
        "stsb": stsb_pair_score_train,
        "quora": quora_pair_train,
        "natural-questions": natural_questions_train,
    }
    
    logger.info(f"Loaded {len(train_dataset)} training datasets")
    for name, dataset in train_dataset.items():
        logger.info(f"  - {name}: {len(dataset)} samples")
    
    # 3. Load evaluation datasets
    # (anchor, positive, negative)
    all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    
    # (sentence1, sentence2, score)
    stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
    
    # (anchor, positive)
    quora_pair_dev = load_dataset(
        "sentence-transformers/quora-duplicates", "pair", 
        split="train[10000:11000]"
    )
    
    # (query, answer)
    natural_questions_dev = load_dataset(
        "sentence-transformers/natural-questions", 
        split="train[10000:11000]"
    )
    
    # Use a dictionary for the evaluation dataset too
    eval_dataset = {
        "all-nli-triplet": all_nli_triplet_dev,
        "stsb": stsb_pair_score_dev,
        "quora": quora_pair_dev,
        "natural-questions": natural_questions_dev,
    }
    
    logger.info(f"Loaded {len(eval_dataset)} evaluation datasets")
    
    # 4. Create loss functions
    # (anchor, positive), (anchor, positive, negative)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model)
    
    # (sentence_A, sentence_B) + class
    softmax_loss = losses.SoftmaxLoss(model)
    
    # (sentence_A, sentence_B) + score
    cosent_loss = losses.CoSENTLoss(model)
    
    # Create mapping of dataset names to loss functions
    base_losses = {
        "all-nli-pair": mnrl_loss,
        "all-nli-pair-class": softmax_loss,
        "all-nli-pair-score": cosent_loss,
        "all-nli-triplet": mnrl_loss,
        "stsb": cosent_loss,
        "quora": mnrl_loss,
        "natural-questions": mnrl_loss,
    }
    
    # Apply Matryoshka wrapper if requested
    if use_matryoshka:
        final_losses = {}
        for dataset_name, loss_fn in base_losses.items():
            final_losses[dataset_name] = losses.MatryoshkaLoss(model, loss_fn, matryoshka_dims=matryoshka_dims)
        logger.info("Applied Matryoshka loss wrapper to all losses")
    else:
        final_losses = base_losses
    
    # 5. Create evaluators for STS benchmark
    sts_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=stsb_pair_score_dev["sentence1"],
        sentences2=stsb_pair_score_dev["sentence2"],
        scores=stsb_pair_score_dev["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev"
    )
    
    # If using Matryoshka, create evaluators for each dimension
    if use_matryoshka:
        evaluators = []
        for dim in matryoshka_dims:
            evaluators.append(
                EmbeddingSimilarityEvaluator(
                    sentences1=stsb_pair_score_dev["sentence1"],
                    sentences2=stsb_pair_score_dev["sentence2"],
                    scores=stsb_pair_score_dev["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name=f"sts-dev-{dim}",
                    truncate_dim=dim,
                )
            )
        dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])
    else:
        dev_evaluator = sts_evaluator
    
    # 6. Define training arguments
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
        seed=args.seed,
        # Tracking parameters:
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=args.eval_steps // 2,
        run_name=f"hybrid-{model_name.split('/')[-1]}",
    )
    
    # 7. Create trainer and start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=final_losses,
        evaluator=dev_evaluator,
    )
    
    trainer.train()
    
    # 8. Final evaluation
    logger.info("Performing final evaluation on STS benchmark test set")
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    
    if use_matryoshka:
        # Create evaluators for each dimension
        test_evaluators = []
        for dim in matryoshka_dims:
            test_evaluators.append(
                EmbeddingSimilarityEvaluator(
                    sentences1=test_dataset["sentence1"],
                    sentences2=test_dataset["sentence2"],
                    scores=test_dataset["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name=f"sts-test-{dim}",
                    truncate_dim=dim,
                )
            )
        test_evaluator = SequentialEvaluator(test_evaluators)
    else:
        # Single evaluator for full dimension
        test_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-test"
        )
    
    test_results = test_evaluator(model)
    
    # 9. Save the final model
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)
    logger.info(f"Model saved to {final_output_dir}")
    
    # 10. (Optional) Push to the Hugging Face Hub
    if args.push_to_hub:
        try:
            hub_model_id = args.hub_model_id
            if not hub_model_id:
                suffix = "-matryoshka" if use_matryoshka else "-hybrid"
                hub_model_id = f"{model_name.split('/')[-1]}{suffix}"
            
            model.push_to_hub(hub_model_id)
            logger.info(f"Model pushed to the Hugging Face Hub: {hub_model_id}")
        except Exception as e:
            logger.error(f"Error uploading model to the Hugging Face Hub: {str(e)}")
            logger.info(
                f"To upload it manually, run:\n"
                f"  model = SentenceTransformer('{final_output_dir}')\n"
                f"  model.push_to_hub('{hub_model_id}')"
            )
    
    return model, test_results


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Train the model
    model, results = train_hybrid_model(args, config)
    
    # Print final results
    logger.info("=== Final Evaluation Results ===")
    for name, score in results.items():
        logger.info(f"{name}: {score:.4f}")
