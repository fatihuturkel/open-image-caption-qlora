# GIT-QLoRA: Fine-tuning GIT Models for Image Captioning

This project implements fine-tuning of Microsoft's Generative Image-to-Text (GIT) models using Quantized Low-Rank Adaptation (QLoRA) for efficient training on image captioning tasks.

## Overview

GIT is a powerful vision-language model that can generate textual descriptions from images. This implementation uses Parameter-Efficient Fine-Tuning (PEFT) with QLoRA to fine-tune GIT models on custom image captioning datasets with minimal computational resources.

## Features

- **Efficient Fine-tuning**: Uses 4-bit quantization and LoRA for memory-efficient training
- **Customizable**: Supports various configurations for model size, LoRA parameters, and training settings
- **Evaluation**: Includes metrics like BLEU, ROUGE, and Fréchet GTE Distance
- **Inference**: Easy-to-use inference pipeline for captioning new images
- **Weight Merging**: Option to merge LoRA weights with the base model for deployment

## Installation

```bash
# Install required packages
pip install transformers peft accelerate datasets bitsandbytes torchvision torch pillow tqdm evaluate rouge_score nltk tensorboard
```

## Dataset Preparation

The dataset should be organized with images in a directory structure and captions either in a CSV file or generated from filenames.

Example directory structure:
```
/data
  /train
    image1.jpg
    image2.jpg
    ...
  /test
    image1.jpg
    image2.jpg
    ...
  captions.csv  # Optional: image_name,caption
```

## Usage

### Training

```bash
python git_qlora_main.py --train \
  --train_dir ./train \
  --test_dir ./test \
  --captions_file ./captions.csv \
  --model_name microsoft/git-base \
  --output_dir ./results \
  --batch_size 4 \
  --epochs 3
```

### Inference

```bash
python git_qlora_main.py --inference \
  --test_dir ./test \
  --output_dir ./results/your_training_run
```

### Single Image Inference

```bash
python git_qlora_main.py --inference \
  --test_image ./test/image.jpg \
  --output_dir ./results/your_training_run
```

### Evaluation

```bash
python git_qlora_main.py --eval \
  --test_dir ./test \
  --captions_file ./test_captions.csv \
  --output_dir ./results/your_training_run
```

### Merging LoRA Weights

```bash
python git_qlora_main.py --merge_weights \
  --output_dir ./results/your_training_run
```

## Configuration Options

The main script supports various configuration options:

- **Dataset**: `--train_dir`, `--test_dir`, `--captions_file`, `--split_ratio`
- **Model**: `--model_name`, `--output_dir`
- **QLoRA**: `--quantization_bits`, `--lora_r`, `--lora_alpha`, `--lora_dropout`
- **Training**: `--batch_size`, `--epochs`, `--learning_rate`, `--gradient_accumulation`
- **Modes**: `--train`, `--eval`, `--inference`, `--merge_weights`

## Project Structure

```
/
├── git_qlora_main.py         # Main script
├── /models
│   └── git_qlora.py          # Model setup with QLoRA
├── /scripts
│   ├── trainer.py            # Training pipeline
│   └── inference.py          # Inference and weight merging
├── /utils
│   ├── dataset_utils.py      # Dataset preparation
│   └── evaluation.py         # Evaluation metrics
├── /results                  # Output directory
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory (4-bit quantization helps reduce requirements)

### Running without GPU

For CPU-only environments:
- Modify `models/git_qlora.py` to disable 4-bit quantization when no GPU is available
- Use smaller batch sizes and model variants
- Consider using `microsoft/git-small` for lower memory requirements
- Be aware that inference and training will be significantly slower

## Acknowledgements

This implementation is based on the following resources:
- [Microsoft GIT](https://huggingface.co/microsoft/git-base) model
- [PEFT](https://github.com/huggingface/peft) library for parameter-efficient fine-tuning
- [Transformers](https://github.com/huggingface/transformers) library
- [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
