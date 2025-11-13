CodeReview-ML: Fine-Tuning CodeLlama-7B for Code Review
This project fine-tunes the codellama/CodeLlama-7b-Instruct-hf model to generate helpful code review comments. It is designed to run in a Google Colab environment and uses QLoRA for memory-efficient training.

The model is trained on the fasterinnerlooper/codereviewer dataset to learn the relationship between a code diff (a "patch") and an associated review comment (a "message").

Tech Stack
Model: codellama/CodeLlama-7b-Instruct-hf

Dataset: fasterinnerlooper/codereviewer (using the train_generation config)

Technique: QLoRA (4-bit Quantization + LoRA)

Framework: PyTorch

Libraries: transformers, peft, trl (SFTTrainer), bitsandbytes, datasets, evaluate

Project Structure
codereviewml/
├── train.py          # Main script to run training and evaluation
├── config.py         # All hyperparameters, paths, and model names
├── data_utils.py     # Data loading and preprocessing logic
├── evaluate.py       # BLEU score evaluation logic
├── utils.py          # Helper functions (checkpointing, GDrive saving)
└── requirements.txt  # Python package requirements
Setup & Installation
This project is built for Google Colab and requires a GPU runtime (T4, A100, etc.).

Open Colab: Start a new Colab notebook, go to Runtime > Change runtime type, and select a T4 GPU (or better).

Open a Terminal: In the Colab UI, click the >_ icon in the bottom-left corner to open a terminal.

Create Project Directory:

Bash

mkdir /content/codereviewml
cd /content/codereviewml
Upload Files: Upload all your Python files (train.py, config.py, etc.) into the /content/codereviewml directory.

Create requirements.txt: Create the requirements file by running this command in your terminal:

Bash

cat << EOF > requirements.txt
transformers
peft
trl
datasets
bitsandbytes
accelerate
evaluate
sacrebleu
torch>=2.0.0
tqdm
EOF
Install Dependencies: This is a critical step. Modern Colab runtimes have pre-installed packages (like sentence-transformers) that conflict with this stack. The following commands will first remove the conflicting package and then install the correct, modern libraries.

Bash

# CRITICAL: Uninstall the conflicting package
pip uninstall -y sentence-transformers

# Install all the required libraries
pip install -r requirements.txt
How to Run
After setting up the environment and installing the dependencies, you can start the entire pipeline with a single command:

Bash

python train.py
The script will:

Log into the terminal.

Download the codellama model and the codereviewer dataset.

Preprocess the 10,000 samples.

Start the SFTTrainer training process, showing progress.

Save checkpoints locally to ./codereview-lora.

Run a baseline (zero-shot) BLEU score evaluation.

Run a final fine-tuned BLEU score evaluation.

Print the final comparison and run a quick demo.

Attempt to save the final LoRA adapter to your Google Drive at /content/drive/MyDrive/codereview-lora-final.

How It Works
config.py: This file defines all static variables. The most important are DATASET_NAME = "fasterinnerlooper/codereviewer" and DATASET_CONFIG_NAME = "train_generation".

data_utils.py: This file loads the data and formats it into the prompt. It correctly uses the patch column for the code diff and the msg column for the review comment.

evaluate.py: This file runs the model against the test set and uses sacrebleu to calculate the BLEU score, comparing the model's generated text to the reference msg from the dataset.

train.py: This script orchestrates the entire process, including fixing API mismatches (like moving max_seq_length to the AutoTokenizer) that were necessary to run a modern transformers stack.

Expected Results
After the script finishes, you will see a final report in your terminal, similar to this:

--- Evaluation Results ---
Zero-Shot BLEU:   [Baseline BLEU score]
Fine-Tuned BLEU:  [Fine-tuned BLEU score]
Improvement:      [...% improvement]
