import os
import argparse
from PIL import Image
import torch
from scripts.inference import GITCaptioner

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned GIT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or adapter")
    parser.add_argument("--base_model", type=str, default="microsoft/git-base", help="Path to the base model if using adapter")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the result")
    parser.add_argument("--use_adapter", action="store_true", help="Use separate adapter instead of merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    if args.use_adapter:
        print(f"Loading base model {args.base_model} with adapter {args.model_path}")
        captioner = GITCaptioner(
            model_path=args.base_model,
            peft_model_path=args.model_path,
            device=device
        )
    else:
        print(f"Loading merged model from {args.model_path}")
        captioner = GITCaptioner(
            model_path=args.model_path,
            device=device
        )
    
    # Open the image
    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Generate caption
    caption = captioner.generate_caption(
        image=image,
        max_length=50,
        num_beams=5,
        temperature=1.0,
    )
    
    print(f"\nImage: {args.image_path}")
    print(f"Caption: {caption}")
    
    # Save result if output path is provided
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Caption: {caption}\n")
        print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()
