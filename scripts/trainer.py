import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoModelForCausalLM,
    AutoProcessor,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, List, Optional, Union, Any
import evaluate
from datasets import DatasetDict
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

# Try to download nltk data, handle case where it's already downloaded
try:
    nltk.download('punkt', quiet=True)
except:
    pass

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics for the model.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    metric_rouge = evaluate.load("rouge")
    metric_bleu = evaluate.load("bleu")
    
    preds, labels = eval_preds
    
    # Replace -100 (padding token) with the pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Prepare inputs for BLEU
    predictions_tokens = [word_tokenize(pred) for pred in pred_str]
    references_tokens = [[word_tokenize(label)] for label in label_str]
    
    # Calculate metrics
    rouge_output = metric_rouge.compute(
        predictions=pred_str, 
        references=label_str,
        use_stemmer=True,
    )
    
    bleu_output = metric_bleu.compute(
        predictions=predictions_tokens,
        references=references_tokens,
    )
    
    # Combine metrics
    metrics = {
        "bleu": bleu_output["bleu"] * 100,  # Convert to percentage
    }
    
    # Add ROUGE scores
    for k, v in rouge_output.items():
        metrics[k] = v * 100  # Convert to percentage
    
    return metrics

class GITTrainer:
    """
    Trainer class for fine-tuning a GIT model with QLoRA.
    """
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        eval_dataset=None,
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        fp16=True,
        bf16=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            processor: The processor (tokenizer + image processor)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device for training
            per_device_eval_batch_size: Batch size per device for evaluation
            gradient_accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Ratio of warmup steps
            logging_steps: Number of steps between logging
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            fp16: Whether to use fp16 mixed precision
            bf16: Whether to use bf16 mixed precision
            evaluation_strategy: Evaluation strategy
            save_strategy: Saving strategy
            load_best_model_at_end: Whether to load the best model at the end
            metric_for_best_model: Metric for determining the best model
            greater_is_better: Whether higher is better for the metric
        """
        self.model = model
        self.processor = processor
        
        # Set the tokenizer for metrics computation
        global tokenizer
        tokenizer = processor.tokenizer
        
        # Set up data collator
        self.data_collator = DataCollatorForSeq2Seq(
            processor.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if fp16 or bf16 else None,
        )
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            fp16=fp16,
            bf16=bf16,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            push_to_hub=False,
            remove_unused_columns=False,  # Important for custom datasets
        )
        
        # Set up the trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
        )
    
    def train(self):
        """
        Train the model and return training results.
        """
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.trainer.save_state()
        
        # Log and return metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        return metrics
    
    def evaluate(self):
        """
        Evaluate the model and return evaluation metrics.
        """
        if self.trainer.eval_dataset is None:
            print("No evaluation dataset provided, skipping evaluation")
            return None
        
        print("Evaluating model...")
        metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def save_model(self, output_dir=None):
        """
        Save the model, either as a complete model or just the LoRA adapter.
        
        Args:
            output_dir: Directory to save the model to (defaults to trainer's output_dir)
            
        Returns:
            Path to the saved model
        """
        if output_dir is None:
            output_dir = os.path.join(self.training_args.output_dir, "final_model")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return output_dir
