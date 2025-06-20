import os
import torch
from PIL import Image
from typing import List, Union, Dict, Any, Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import time

class GITCaptioner:
    """
    Class for inference with a fine-tuned GIT model.
    """
    def __init__(
        self,
        model_path: str,
        peft_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the captioner.
        
        Args:
            model_path: Path to the base model or merged model
            peft_model_path: Path to the PEFT adapter (if not merged)
            device: Device to run inference on
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Loading model from {model_path}")
        
        # If we have a separate PEFT adapter, load it
        if peft_model_path is not None:
            print(f"Loading base model and applying PEFT adapter from {peft_model_path}")
            # Load the base model without quantization for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                device_map=device,
            )
            
            # Load the PEFT adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_model_path,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                device_map=device,
            )
        else:
            # Load the merged model directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                device_map=device,
            )
        
        # Set the model to evaluation mode
        self.model.eval()
        print(f"Model loaded on {device}")
    
    def generate_caption(
        self,
        image: Union[str, Image.Image],
        max_length: int = 50,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: Path to image or PIL Image object
            max_length: Maximum length of the generated caption
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters
            
        Returns:
            Generated caption
        """
        # Load the image if a path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate the caption
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        # Decode the caption
        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Print generation time
        generation_time = time.time() - start_time
        print(f"Caption generated in {generation_time:.2f} seconds")
        
        return caption
    
    def batch_generate_captions(
        self,
        images: List[Union[str, Image.Image]],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for a batch of images.
        
        Args:
            images: List of image paths or PIL Image objects
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated captions
        """
        captions = []
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Load images if paths are provided
            loaded_batch = []
            for img in batch:
                if isinstance(img, str):
                    loaded_batch.append(Image.open(img).convert("RGB"))
                else:
                    loaded_batch.append(img)
            
            # Process the batch
            inputs = self.processor(images=loaded_batch, return_tensors="pt", padding=True).to(self.device)
            
            # Generate the captions
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
            
            # Decode the captions
            batch_captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            captions.extend(batch_captions)
        
        return captions

def merge_lora_weights(
    base_model_path: str,
    peft_model_path: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Merge LoRA weights with the base model.
    
    Args:
        base_model_path: Path to the base model
        peft_model_path: Path to the PEFT adapter
        output_path: Path to save the merged model
        device: Device to use for merging
    """
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device,
    )
    
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    print(f"Loading PEFT adapter from {peft_model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map=device,
    )
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    print("Model merged and saved successfully")
