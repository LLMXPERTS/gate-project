"""
Utility functions for model setup and management.
"""

import logging
import os
from typing import Optional, Union, Dict, Any

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def setup_model(
    model_name_or_path: str,
    max_seq_length: int = 128,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> SentenceTransformer:
    """
    Set up a SentenceTransformer model for training or evaluation.
    
    Args:
        model_name_or_path: Model name or path
        max_seq_length: Maximum sequence length for tokenization
        device: Device to load the model on (auto-detected if None)
        cache_dir: Directory to cache models
    
    Returns:
        Loaded SentenceTransformer model
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    logger.info(f"Loading model from {model_name_or_path} on {device}")
    model = SentenceTransformer(model_name_or_path, device=device, cache_folder=cache_dir)
    
    # Set max sequence length
    model.max_seq_length = max_seq_length
    logger.info(f"Model max sequence length set to {max_seq_length}")
    
    return model


def save_model(
    model: SentenceTransformer, 
    output_path: str, 
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_private: bool = False,
    hub_token: Optional[str] = None,
    save_extra_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a trained model locally and optionally push to Hugging Face Hub.
    
    Args:
        model: SentenceTransformer model to save
        output_path: Path to save the model
        push_to_hub: Whether to push the model to the Hugging Face Hub
        hub_model_id: Model ID for Hugging Face Hub (defaults to output_path basename)
        hub_private: Whether the model should be private on the Hub
        hub_token: Authentication token for Hugging Face Hub
        save_extra_info: Additional information to save with the model
    
    Returns:
        Path where the model was saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save extra information if provided
    if save_extra_info:
        model.save_to_hub(
            repo_name=output_path.split("/")[-1],
            model_name_or_path=output_path,
            organization=None,  # Set to organization name if needed
            private=False,
            exist_ok=True,
            replace_model_card=True,
            train_datasets=save_extra_info.get("train_datasets", []),
        )
    else:
        # Standard save
        model.save(output_path)
    
    logger.info(f"Model saved to {output_path}")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub:
        if hub_model_id is None:
            hub_model_id = os.path.basename(output_path)
        
        try:
            model.push_to_hub(
                repo_name=hub_model_id,
                private=hub_private,
                use_auth_token=hub_token
            )
            logger.info(f"Model pushed to Hugging Face Hub: {hub_model_id}")
        except Exception as e:
            logger.error(f"Error pushing model to hub: {str(e)}")
    
    return output_path


def load_model_with_dimension(
    model_name_or_path: str,
    dimension: Optional[int] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Union[SentenceTransformer, 'TruncatedModel']:
    """
    Load a SentenceTransformer model with optional dimension truncation for
    Matryoshka models.
    
    Args:
        model_name_or_path: Model name or path
        dimension: Dimension to truncate embeddings to (if provided)
        device: Device to load the model on
        cache_dir: Directory to cache models
    
    Returns:
        Loaded model (with truncation wrapper if dimension is specified)
    """
    # Load the base model
    model = setup_model(model_name_or_path, device=device, cache_dir=cache_dir)
    
    # If dimension is provided, create a wrapper that truncates embeddings
    if dimension is not None:
        class TruncatedModel:
            def __init__(self, base_model, dim):
                self.model = base_model
                self.truncate_dim = dim
                self.embedding_dimension = dim
            
            def encode(self, sentences, batch_size=32, **kwargs):
                embeddings = self.model.encode(sentences, batch_size=batch_size, **kwargs)
                return embeddings[:, :self.truncate_dim]
        
        logger.info(f"Creating truncated model with dimension {dimension}")
        return TruncatedModel(model, dimension)
    
    return model


def get_model_info(model: SentenceTransformer) -> Dict[str, Any]:
    """
    Get information about a SentenceTransformer model.
    
    Args:
        model: SentenceTransformer model
    
    Returns:
        Dictionary with model information
    """
    info = {
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "modules": str(model),
        "device": next(model.parameters()).device.type
    }
    
    # Try to get model architecture
    if hasattr(model, "_modules") and len(model._modules) > 0:
        first_module = next(iter(model._modules.values()))
        if hasattr(first_module, "auto_model"):
            info["architecture"] = first_module.auto_model.config.model_type
    
    return info
