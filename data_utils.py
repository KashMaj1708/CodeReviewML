"""
Data utilities for loading, processing, and formatting the CodeReviewer dataset.
"""

import logging
from datasets import load_dataset
import config

def create_diff(sample):
    """
    Generates a simple text-based diff as specified ("just concatenate").
    """
    return f"--- BEFORE ---\n{sample['code_before']}\n\n--- AFTER ---\n{sample['code_after']}"

def format_for_sft(sample):
    """
    Formats a single data sample into the required "text" field for SFTTrainer.
    """
    diff_text = create_diff(sample)
    full_prompt = config.PROMPT_TEMPLATE.format(diff=diff_text)
    
    # The 'text' field must contain the full prompt + response
    return {"text": full_prompt + sample['review_comment']}

def get_datasets():
    """
    Loads, splits, and formats the dataset.
    
    Returns:
        tuple: (formatted_train_dataset, formatted_eval_dataset_for_trainer, raw_eval_dataset_for_bleu)
    """
    logging.info(f"Loading dataset: {config.DATASET_NAME} (first {config.NUM_SAMPLES} samples)")
    dataset = load_dataset(config.DATASET_NAME, split=f"train[:{config.NUM_SAMPLES}]")
    
    # Split: 9000 train / 1000 eval (as per spec)
    split_dataset = dataset.train_test_split(
        test_size=(1.0 - config.TRAIN_TEST_SPLIT_RATIO), 
        shuffle=True, 
        seed=42
    )
    raw_train_dataset = split_dataset["train"]
    raw_eval_dataset = split_dataset["test"] # This is our held-out test set
    
    logging.info(f"Data split: {len(raw_train_dataset)} train, {len(raw_eval_dataset)} eval")

    # Format datasets for SFTTrainer
    train_dataset = raw_train_dataset.map(format_for_sft)
    eval_dataset_for_trainer = raw_eval_dataset.map(format_for_sft)
    
    logging.info("Data formatting complete.")
    logging.info("--- Verifying 1 data sample ---")
    logging.info(train_dataset[0]['text'])
    logging.info("---------------------------------")
    
    return train_dataset, eval_dataset_for_trainer, raw_eval_dataset