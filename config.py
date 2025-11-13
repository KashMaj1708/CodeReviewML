"""
Configuration file for the CodeReview-ML project.
All static variables, hyperparameters, and paths are defined here.
"""

import torch

# --- Model ---
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

# --- Data ---
DATASET_NAME = "microsoft/codereview"
NUM_SAMPLES = 10000  # "10K+ review pairs"
TRAIN_TEST_SPLIT_RATIO = 0.9
MAX_SEQ_LENGTH = 512 # Shorter = faster

# --- Prompt Template ---
PROMPT_TEMPLATE = """### Code Change:
{diff}

### Review Comment:
"""

# --- LoRA ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Training ---
OUTPUT_DIR = "./codereview-lora"
EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 8   # Adjust to 4 if OOM
GRAD_ACCUMULATION_STEPS = 4 # Effective batch size = 32
LEARNING_RATE = 2e-4
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
EVAL_STEPS = 250
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
FP16 = not BF16

# --- Evaluation ---
EVAL_MAX_NEW_TOKENS = 150 # For generating review comments
EVAL_TEMPERATURE = 0.7
EVAL_DO_SAMPLE = True

# --- Google Drive ---
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_PATH = "/content/drive/MyDrive/codereview-lora-final"