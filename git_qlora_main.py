import os
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime

# Import our modules
from utils.dataset_utils import load_images_with_captions, prepare_dataset_for_git
from models.git_qlora import load_git_model_for_qlora
from scripts.trainer import GITTrainer
from scripts.inference import GITCaptioner, merge_lora_weights
from utils.evaluation import evaluate_model_on_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a GIT model with QLoRA")
    
    # Dataset arguments
    parser.add_argument("--train_dir", type=str, default="./train", help="Path to the training image directory")
    parser.add_argument("--test_dir", type=str, default="./test", help="Path to the test image directory")
    parser.add_argument("--captions_file", type=str, default=None, help="Path to captions file (CSV)")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/git-base", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    # QLoRA arguments
    parser.add_argument("--quantization_bits", type=int, default=4, choices=[4, 8], help="Quantization bits (4 or 8)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--inference", action="store_true", help="Run inference on test set")
    parser.add_argument("--merge_weights", action="store_true", help="Merge LoRA weights with base model")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    
    # Inference arguments
    parser.add_argument("--test_image", type=str, default=None, help="Path to a single test image for inference")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"git_qlora_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.train:
        print("=== Starting Training Pipeline ===")
        
        # Load dataset
        print(f"Loading dataset from {args.train_dir}")
        dataset = load_images_with_captions(
            image_dir=args.train_dir,
            caption_file=args.captions_file,
            split_ratio=args.split_ratio
        )
        
        print(f"Dataset loaded: {len(dataset['train'])} training examples, "
              f"{len(dataset.get('validation', [])) if 'validation' in dataset else 0} validation examples")
        
        # Load model
        model, processor = load_git_model_for_qlora(
            model_name_or_path=args.model_name,
            quantization_bits=args.quantization_bits,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        
        # Prepare dataset for training
        processed_dataset = prepare_dataset_for_git(dataset, processor)
        
        # Set up trainer
        trainer = GITTrainer(
            model=model,
            processor=processor,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset.get("validation", None),
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
        )
        
        # Train the model
        train_metrics = trainer.train()
        print(f"Training complete. Metrics: {train_metrics}")
        
        # Evaluate the model
        if "validation" in processed_dataset:
            eval_metrics = trainer.evaluate()
            print(f"Evaluation complete. Metrics: {eval_metrics}")
        
        # Save the model
        trainer.save_model()
    
    if args.merge_weights:
        print("=== Merging LoRA Weights with Base Model ===")
        
        # Determine paths
        base_model_path = args.model_name
        peft_model_path = os.path.join(output_dir, "final_model")
        merged_model_path = os.path.join(output_dir, "merged_model")
        
        # Merge weights
        merge_lora_weights(
            base_model_path=base_model_path,
            peft_model_path=peft_model_path,
            output_path=merged_model_path,
            device=device,
        )
    
    if args.inference:
        print("=== Running Inference ===")
        
        # Determine model path
        if args.merge_weights:
            model_path = os.path.join(output_dir, "merged_model")
            peft_model_path = None
        else:
            model_path = args.model_name
            peft_model_path = os.path.join(output_dir, "final_model")
        
        # Load captioner
        captioner = GITCaptioner(
            model_path=model_path,
            peft_model_path=peft_model_path,
            device=device,
        )
        
        # Run inference on single image or test directory
        if args.test_image:
            caption = captioner.generate_caption(args.test_image)
            print(f"Caption for {args.test_image}: {caption}")
        elif args.test_dir:
            # List all image files in the test directory
            test_images = []
            for ext in ['.jpg', '.jpeg', '.png']:
                test_images.extend(list(Path(args.test_dir).glob(f"*{ext}")))
            
            print(f"Running inference on {len(test_images)} test images")
            
            # Generate captions in batches
            captions = captioner.batch_generate_captions(
                images=[str(img) for img in test_images],
                batch_size=args.batch_size,
            )
            
            # Save results
            results = []
            for img, caption in zip(test_images, captions):
                results.append({
                    "image": str(img),
                    "caption": caption,
                })
            
            # Save results to file
            results_file = os.path.join(output_dir, "inference_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Inference results saved to {results_file}")
    
    if args.eval and not args.train:
        print("=== Running Standalone Evaluation ===")
        
        # Determine model path
        if os.path.exists(os.path.join(output_dir, "merged_model")):
            model_path = os.path.join(output_dir, "merged_model")
            peft_model_path = None
        else:
            model_path = args.model_name
            peft_model_path = os.path.join(output_dir, "final_model")
        
        # Load captioner (which loads the model)
        captioner = GITCaptioner(
            model_path=model_path,
            peft_model_path=peft_model_path,
            device=device,
        )
        
        # Load test dataset
        test_dataset = load_images_with_captions(
            image_dir=args.test_dir,
            caption_file=args.captions_file,
            split_ratio=0.0  # No need to split for evaluation
        )["train"]  # Use the whole dataset
        
        # Evaluate
        metrics = evaluate_model_on_dataset(
            model=captioner.model,
            processor=captioner.processor,
            dataset=test_dataset,
            batch_size=args.batch_size,
            device=device,
        )
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w") as f:
            # Handle non-serializable items
            metrics_serializable = {k: v for k, v in metrics.items() if k != "examples"}
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Evaluation metrics: {metrics}")
        print(f"Evaluation metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
