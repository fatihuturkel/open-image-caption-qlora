import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a GIT model")
    parser.add_argument("--model_name", type=str, default="microsoft/git-base", help="Model name or path")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the result")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the processor and model
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Open the image
    try:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Image loaded: {args.image_path}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption
    print("Generating caption...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
        )
    inference_time = time.time() - start_time
    
    # Decode the output
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"\nImage: {args.image_path}")
    print(f"Caption: {caption}")
    print(f"Generated in {inference_time:.2f} seconds")
    
    # Save result if output path is provided
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Caption: {caption}\n")
            f.write(f"Generated in {inference_time:.2f} seconds\n")
        print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()
