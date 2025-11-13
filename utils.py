"""
Utility functions for checkpointing, saving to Drive, and environment checks.
"""

import logging
import torch
from pathlib import Path
import config

def check_bf16_support():
    """Logs whether bfloat16 is supported."""
    if config.BF16:
        logging.info("Using bf16 (A100/H100/RTX 30xx+ detected)")
    else:
        logging.info("Using fp16 (bf16 not available)")

def find_latest_checkpoint(directory: str) -> str | None:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    logging.info(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)

def save_to_drive(trainer, tokenizer):
    """Mounts Google Drive and saves the final model and tokenizer."""
    try:
        from google.colab import drive
        logging.info(f"Mounting Google Drive at {config.DRIVE_MOUNT_PATH}...")
        drive.mount(config.DRIVE_MOUNT_PATH)
        
        save_path = Path(config.DRIVE_SAVE_PATH)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving final model adapter to {save_path}...")
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        logging.info("Model saved to Google Drive successfully.")

    except ImportError:
        logging.warning("Google Colab not detected. Skipping Google Drive save.")
    except Exception as e:
        logging.error(f"Error saving to Google Drive: {e}")