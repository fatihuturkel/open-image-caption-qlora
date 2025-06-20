import os
import torch
from PIL import Image
import time
from transformers import AutoProcessor, AutoModelForCausalLM

def test_model_setup():
    """Test that the model loads correctly"""
    print("Testing model setup...")
    
    # First, try loading the model without quantization
    try:
        print("Loading model without quantization (for testing only)...")
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
        print("✅ Model loaded successfully!")        
        # Print some model info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        print(f"Total parameters: {total_params:,}")
        
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def test_image_processing(processor):
    """Test that images can be processed"""
    print("\nTesting image processing...")
    
    # Find a test image
    test_dir = "test"
    if os.path.exists(test_dir):
        image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image_path = os.path.join(test_dir, image_files[0])
            print(f"Using test image: {test_image_path}")
            
            try:
                # Load and process the image
                image = Image.open(test_image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                print(f"✅ Image processed successfully!")
                print(f"Image tensor shape: {inputs.pixel_values.shape}")
                return inputs
            except Exception as e:
                print(f"❌ Failed to process image: {e}")
                return None
        else:
            print("❌ No image files found in test directory")
            return None
    else:
        print("❌ Test directory not found")
        return None

def test_inference(model, processor, inputs):
    """Test that the model can generate captions"""
    print("\nTesting inference...")
    
    if model is None or inputs is None:
        print("❌ Cannot run inference due to previous errors")
        return
    
    try:
        # Set model to eval mode
        model.eval()
        
        # Run inference
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
        
        print(f"✅ Inference completed in {inference_time:.2f} seconds")
        print(f"Generated caption: {caption}")
    except Exception as e:
        print(f"❌ Failed to run inference: {e}")

def main():
    print("=== GIT-QLoRA Setup Test ===")
    
    # Test model setup
    model, processor = test_model_setup()
    
    # Test image processing
    inputs = test_image_processing(processor)
    
    # Test inference
    test_inference(model, processor, inputs)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
