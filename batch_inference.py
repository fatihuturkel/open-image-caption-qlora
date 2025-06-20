import os
import argparse
import json
from pathlib import Path
import torch
from scripts.inference import GITCaptioner

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference with a fine-tuned GIT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or adapter")
    parser.add_argument("--base_model", type=str, default="microsoft/git-base", help="Path to the base model if using adapter")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output_file", type=str, default="caption_results.json", help="Path to save the results")
    parser.add_argument("--use_adapter", action="store_true", help="Use separate adapter instead of merged model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
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
    
    # Collect all image paths
    test_dir = Path(args.test_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(test_dir.glob(ext)))
    
    # Limit the number of images if specified
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} images in {args.test_dir}")
    
    # Generate captions in batches
    image_paths_str = [str(path) for path in image_paths]
    captions = captioner.batch_generate_captions(
        images=image_paths_str,
        batch_size=args.batch_size,
        max_length=50,
        num_beams=5,
    )
    
    # Create results
    results = []
    for img_path, caption in zip(image_paths_str, captions):
        results.append({
            "image": img_path,
            "caption": caption
        })
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")
    
    # Print a few examples
    print("\nExample captions:")
    for i, result in enumerate(results[:5]):
        print(f"Image: {result['image']}")
        print(f"Caption: {result['caption']}")
        print()

if __name__ == "__main__":
    main()
