"""
Evaluation logic for computing BLEU scores against a test set.
"""

import logging
import evaluate
from tqdm import tqdm
import config
from data_utils import create_diff

def run_bleu_evaluation(model, tokenizer, dataset):
    """
    Runs generation on the test set and computes BLEU score.
    
    Args:
        model: The model to evaluate (either base or fine-tuned).
        tokenizer: The corresponding tokenizer.
        dataset: The *raw* evaluation dataset (not formatted for SFT).
    """
    logging.info("Starting BLEU evaluation...")
    bleu_metric = evaluate.load("sacrebleu")
    predictions = []
    references = []

    # Iterate over the raw dataset
    for sample in tqdm(dataset, desc="Evaluating BLEU"):
        # 1. Format the prompt (input only)
        diff_text = create_diff(sample)
        prompt = config.PROMPT_TEMPLATE.format(diff=diff_text)
        
        # 2. Tokenize and truncate
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=config.MAX_SEQ_LENGTH, 
            truncation=True
        ).to(model.device)

        # 3. Generate the review comment
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
            temperature=config.EVAL_TEMPERATURE,
            do_sample=config.EVAL_DO_SAMPLE,
        )
        
        # 4. Decode and extract the new text
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part after the prompt
        prompt_marker = "### Review Comment:"
        pred_text = full_text.split(prompt_marker)[-1].strip()

        # 5. Add to lists
        predictions.append(pred_text)
        references.append([sample['review_comment']]) # sacreBLEU expects list of lists

    # Compute and return score
    logging.info("BLEU evaluation complete.")
    return bleu_metric.compute(predictions=predictions, references=references)