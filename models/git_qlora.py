import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from typing import Dict, Optional, List, Union

def load_git_model_for_qlora(
    model_name_or_path: str = "microsoft/git-base",
    quantization_bits: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device_map: str = "auto"
):
    """
    Load a GIT model and prepare it for QLoRA fine-tuning.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        quantization_bits: Number of bits for quantization (4 or 8)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout probability
        target_modules: List of module names to apply LoRA to
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading GIT model: {model_name_or_path}")
    
    # Load the processor first
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    # Check if GPU is available
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        # Configure quantization for GPU
        compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=quantization_bits == 4,
            load_in_8bit=quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load the model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            device_map=device_map,
        )
        
        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # For CPU-only environments, load without quantization
        print("CUDA not available. Loading model without quantization.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=None,  # Don't use device_map for CPU
        )
        # Note: Without prepare_model_for_kbit_training since we're not using quantization
    
    # If target modules are not specified, use default modules for GIT
    if target_modules is None:
        # GIT is based on T5, so we target typical transformer attention modules
        target_modules = [
            "q", "k", "v", "o",  # Attention modules
            "wi", "wo",          # Feed-forward modules
        ]
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    
    # Print some info about the model
    print(f"Model loaded and prepared for QLoRA fine-tuning")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, processor


# For importing in other files
from transformers import BitsAndBytesConfig
