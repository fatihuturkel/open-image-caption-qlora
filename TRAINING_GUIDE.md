# Step-by-Step Guide to Training GIT with QLoRA

This document provides a step-by-step guide to training a GIT (Generative Image-to-Text) model using QLoRA fine-tuning on your custom dataset.

## 1. Environment Setup

First, ensure you have the necessary environment set up:

```bash
# Install required packages
pip install -r requirements.txt
```

For best performance, use a machine with a CUDA-capable GPU. If running on CPU only, be aware that training will be significantly slower.

## 2. Dataset Preparation

1. **Organize your images**:
   - Place your training images in the `train/` directory
   - Place your test/validation images in the `test/` directory

2. **Prepare captions** (choose one method):
   - **Option A - Create a captions file**: Create a CSV file with image names and captions:
     ```
     image1.jpg,A beautiful sunset over mountains
     image2.jpg,A dog running in the park
     ```
   - **Option B - Use automatic captions**: If no captions file is provided, the system will use filenames as basic captions.

## 3. Quick Start Training

For a simple training run with default parameters:

```bash
python quick_train.py --train_dir ./train --captions_file ./captions.csv --epochs 3
```

This will:
1. Load the microsoft/git-base model
2. Apply QLoRA with default parameters
3. Train for 3 epochs on your dataset
4. Save the model to ./results/quickstart/final_model

## 4. Full Training Pipeline

For more control over the training process:

```bash
python git_qlora_main.py --train --train_dir ./train --test_dir ./test \
  --captions_file ./captions.csv --split_ratio 0.1 \
  --model_name microsoft/git-base --output_dir ./results \
  --batch_size 4 --epochs 5 --learning_rate 5e-5 \
  --lora_r 16 --lora_alpha 32 --quantization_bits 4
```

This provides full control over:
- Dataset organization and splitting
- Model selection
- Training hyperparameters
- QLoRA configuration

## 5. Inference with Trained Model

After training, you can generate captions for new images:

```bash
# For a single image
python run_inference.py --model_path ./results/final_model \
  --base_model microsoft/git-base --image_path ./test/new_image.jpg --use_adapter

# For a batch of images
python batch_inference.py --model_path ./results/final_model \
  --base_model microsoft/git-base --test_dir ./test --output_file captions.json --use_adapter
```

## 6. Merging Weights (Optional)

If you want to create a standalone model without requiring the base model:

```bash
python git_qlora_main.py --merge_weights --output_dir ./results/your_training_run
```

After merging, you can use the model without specifying the base model or adapter:

```bash
python run_inference.py --model_path ./results/your_training_run/merged_model \
  --image_path ./test/new_image.jpg
```

## 7. Evaluation

To evaluate your model on a test set:

```bash
python git_qlora_main.py --eval --test_dir ./test \
  --captions_file ./test_captions.csv --output_dir ./results/your_training_run
```

This will calculate metrics like BLEU and ROUGE scores for your model.

## 8. Alternative for CPU-Only Environment

If you're running on a CPU-only environment, use the simplified inference script:

```bash
python simple_inference.py --model_name microsoft/git-base \
  --image_path ./test/image.jpg
```

## 9. Tips for Best Results

- **Image quality**: Use clear, well-composed images
- **Caption quality**: Provide detailed, accurate captions
- **Training time**: Train for more epochs for better results (monitor validation metrics)
- **Hyperparameters**: Experiment with different learning rates and batch sizes
- **Model size**: Try microsoft/git-base-coco (pre-trained on COCO) for better starting performance
