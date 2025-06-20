import os
import argparse
import torch
from utils.dataset_utils import load_images_with_captions, prepare_dataset_for_git
from models.git_qlora import load_git_model_for_qlora
from scripts.trainer import GITTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train GIT model with QLoRA on a small dataset")
    
    # Dataset arguments
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--captions_file", type=str, default=None, help="Path to captions file (optional)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/git-base", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./results/quickstart", help="Output directory")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    # QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.train_dir}")
    dataset = load_images_with_captions(
        image_dir=args.train_dir,
        caption_file=args.captions_file,
        split_ratio=args.val_split
    )
    
    print(f"Dataset loaded: {len(dataset['train'])} training examples")
    if 'validation' in dataset:
        print(f"{len(dataset['validation'])} validation examples")
    
    # Load model
    print(f"Loading GIT model: {args.model_name}")
    model, processor = load_git_model_for_qlora(
        model_name_or_path=args.model_name,
        quantization_bits=4,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Prepare dataset for training
    print("Processing dataset for training...")
    processed_dataset = prepare_dataset_for_git(dataset, processor)
    
    # Set up trainer
    trainer = GITTrainer(
        model=model,
        processor=processor,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("validation", None),
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # Train the model
    print("Starting training...")
    train_metrics = trainer.train()
    print(f"Training complete. Metrics: {train_metrics}")
    
    # Save the model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")
    
    print("\nTo use this model for inference, run:")
    print(f"python run_inference.py --model_path {os.path.join(args.output_dir, 'final_model')} --base_model {args.model_name} --image_path <your_image> --use_adapter")

if __name__ == "__main__":
    main()
