"""
Main Training Script for CodeReview-ML.

This script orchestrates the entire fine-tuning process:
1. Loads configuration.
2. Sets up models and tokenizers.
3. Prepares the data.
4. Runs the SFTTrainer.
5. Evaluates the model against a baseline.
6. Saves the final model.
"""

import logging
import sys
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Import our custom modules
import config
from data_utils import get_datasets
from evaluate import run_bleu_evaluation
from utils import check_bf16_support, find_latest_checkpoint, save_to_drive

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    start_time = datetime.now()
    logging.info("--- CodeReview ML Fine-Tuning Started ---")

    # --- Phase 1: Setup ---
    logging.info("--- Phase 1: Model and Tokenizer Setup ---")
    check_bf16_support()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    logging.info(f"Loading base model: {config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Phase 2: Data Preparation ---
    logging.info("--- Phase 2: Data Preparation ---")
    train_dataset, eval_dataset_for_trainer, raw_eval_dataset = get_datasets()

    # --- Phase 3: Training ---
    logging.info("--- Phase 3: Training ---")
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        optim=config.OPTIMIZER,
        fp16=config.FP16,
        bf16=config.BF16,
        logging_steps=config.LOGGING_STEPS,
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        evaluation_strategy="steps",
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
    )
    
    latest_checkpoint = find_latest_checkpoint(config.OUTPUT_DIR)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_trainer,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        packing=True,
    )

    logging.info("Starting SFTTrainer.train()...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    logging.info("Training complete. Saving final adapter...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    # --- Phase 4: Evaluation ---
    logging.info("--- Phase 4: Evaluation ---")
    
    logging.info("Reloading base model for Zero-Shot Baseline evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    baseline_results = run_bleu_evaluation(base_model, base_tokenizer, raw_eval_dataset)
    logging.info(f"Zero-Shot Baseline BLEU: {baseline_results['score']:.4f}")
    
    del base_model
    torch.cuda.empty_cache()

    logging.info("Running Fine-Tuned Model evaluation...")
    finetuned_results = run_bleu_evaluation(trainer.model, tokenizer, raw_eval_dataset)
    logging.info(f"Fine-Tuned Model BLEU: {finetuned_results['score']:.4f}")

    baseline_bleu = baseline_results['score']
    finetuned_bleu = finetuned_results['score']
    improvement = ((finetuned_bleu - baseline_bleu) / baseline_bleu) * 100

    logging.info("--- Evaluation Results ---")
    logging.info(f"Zero-Shot BLEU:   {baseline_bleu:.4f}")
    logging.info(f"Fine-Tuned BLEU:  {finetuned_bleu:.4f}")
    logging.info(f"Improvement:      {improvement:.2f}%")
    logging.info("--------------------------")

    # --- Phase 5: Demo & Save ---
    logging.info("--- Phase 5: Demo & Save to Drive ---")

    logging.info("Running quick inference demo...")
    test_diff = """--- BEFORE ---
def get_user(id):
    user = db.query(id)
    return user

--- AFTER ---
def get_user(user_id):
    user = db.query(user_id)
    if user:
        return user
    else:
        return None
"""
    prompt = config.PROMPT_TEMPLATE.format(diff=test_diff)
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    outputs = trainer.model.generate(
        **inputs, max_new_tokens=100, temperature=0.2
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    review = full_text.split("### Review Comment:")[-1].strip()
    
    logging.info("--- DEMO ---")
    logging.info(f"Input Diff:\n{test_diff}\n")
    logging.info(f"Generated Review:\n{review}")
    logging.info("------------")
    
    save_to_drive(trainer, tokenizer)

    end_time = datetime.now()
    logging.info(f"--- Total execution time: {end_time - start_time} ---")


if __name__ == "__main__":
    main()