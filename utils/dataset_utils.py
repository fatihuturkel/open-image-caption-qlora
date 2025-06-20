import os
import glob
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as ImageFeature
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoProcessor
import torch
from torch.utils.data import DataLoader

def load_images_with_captions(
    image_dir: str, 
    captions: Union[Dict[str, str], List[Tuple[str, str]]] = None,
    caption_file: str = None,
    caption_delimiter: str = ",",
    split_ratio: float = 0.1
) -> Dataset:
    """
    Load images and their captions into a HuggingFace Dataset.
    
    Args:
        image_dir: Directory containing images
        captions: Dictionary mapping image filenames to captions or list of (image_path, caption) tuples
        caption_file: Path to a CSV file with image filenames and captions
        caption_delimiter: Delimiter for the caption file
        split_ratio: Ratio of validation set (between 0 and 1)
        
    Returns:
        A DatasetDict with 'train' and 'validation' splits
    """
    image_files = []
    image_captions = []
    
    # If a caption file is provided, read captions from it
    if caption_file and os.path.exists(caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(caption_delimiter, 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(image_dir, img_name)
                    if os.path.exists(img_path):
                        image_files.append(img_path)
                        image_captions.append(caption)
    
    # If captions are provided as a dictionary
    elif isinstance(captions, dict):
        for img_name, caption in captions.items():
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                image_files.append(img_path)
                image_captions.append(caption)
    
    # If captions are provided as a list of tuples
    elif isinstance(captions, list) and all(isinstance(item, tuple) and len(item) == 2 for item in captions):
        for img_path, caption in captions:
            if os.path.exists(img_path):
                image_files.append(img_path)
                image_captions.append(caption)
    
    # If no captions are provided, use dummy captions or generate from filenames
    else:
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(image_dir, ext)))
            all_images.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        
        for img_path in all_images:
            image_files.append(img_path)
            # Use filename as a placeholder caption
            base_name = os.path.basename(img_path).split('.')[0]
            # Convert filename to a more readable format (e.g., "image_01" -> "image 01")
            caption = ' '.join(base_name.split('_'))
            image_captions.append(caption)
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Create a dataset
    dataset = Dataset.from_dict({
        "image_path": image_files,
        "caption": image_captions
    })
    
    # Define a function to load images when needed
    def load_image(example):
        try:
            image = Image.open(example["image_path"]).convert("RGB")
            example["image"] = image
            return example
        except Exception as e:
            print(f"Error loading {example['image_path']}: {e}")
            return None
    
    # Apply the image loading function
    dataset = dataset.map(load_image, remove_columns=["image_path"])
    
    # Split the dataset
    if split_ratio > 0:
        split_dataset = dataset.train_test_split(test_size=split_ratio)
        # Rename 'test' split to 'validation'
        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    else:
        return DatasetDict({
            "train": dataset
        })

def prepare_dataset_for_git(
    dataset: DatasetDict, 
    processor, 
    max_length: int = 512,
):
    """
    Prepare a dataset for training the GIT model.
    
    Args:
        dataset: DatasetDict with 'train' and optional 'validation' splits
        processor: GIT processor (tokenizer + image processor)
        max_length: Maximum token length for captions
        
    Returns:
        Processed datasets ready for training
    """
    def process_examples(examples):
        # Process the images
        pixel_values = processor(
            images=examples["image"], 
            return_tensors="pt",
            padding="max_length",
        ).pixel_values
        
        # Process the captions
        tokenized_captions = processor.tokenizer(
            examples["caption"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized_captions.input_ids,
            "attention_mask": tokenized_captions.attention_mask,
            "labels": tokenized_captions.input_ids.clone(),
        }
    
    # Apply preprocessing to each split
    processed_dataset = {}
    for split in dataset.keys():
        processed_dataset[split] = dataset[split].map(
            process_examples,
            batched=True,
            remove_columns=dataset[split].column_names,
        )
        
        # Set the format for PyTorch
        processed_dataset[split].set_format("torch")
    
    return DatasetDict(processed_dataset)

def create_dataloaders(
    processed_dataset: DatasetDict, 
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 4
):
    """
    Create PyTorch DataLoaders from processed datasets
    
    Args:
        processed_dataset: Processed DatasetDict
        batch_size: Batch size for training
        shuffle_train: Whether to shuffle the training data
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split, dataset in processed_dataset.items():
        is_train = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train and shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders
