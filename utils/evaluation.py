import torch
import numpy as np
import evaluate
from typing import List, Dict, Union, Any
import nltk
from nltk.tokenize import word_tokenize

# Try to download nltk data, handle case where it's already downloaded
try:
    nltk.download('punkt', quiet=True)
except:
    pass

def compute_captioning_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Compute metrics for image captioning evaluation.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        
    Returns:
        Dictionary of metrics
    """
    # Load metrics
    metric_rouge = evaluate.load("rouge")
    metric_bleu = evaluate.load("bleu")
    
    # Prepare inputs for BLEU
    predictions_tokens = [word_tokenize(pred) for pred in predictions]
    references_tokens = [[word_tokenize(ref)] for ref in references]
    
    # Calculate metrics
    rouge_output = metric_rouge.compute(
        predictions=predictions, 
        references=references,
        use_stemmer=True,
    )
    
    bleu_output = metric_bleu.compute(
        predictions=predictions_tokens,
        references=references_tokens,
    )
    
    # Combine metrics
    metrics = {
        "bleu": bleu_output["bleu"] * 100,  # Convert to percentage
    }
    
    # Add ROUGE scores
    for k, v in rouge_output.items():
        metrics[k] = v * 100  # Convert to percentage
    
    return metrics

def calculate_frechet_gte_distance(
    predictions: List[str], 
    references: List[str],
    model_name: str = "thenlper/gte-large"
) -> float:
    """
    Calculate Fréchet GTE Distance (a text embedding-based metric similar to FID).
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        model_name: Name of the sentence embedding model to use
        
    Returns:
        Fréchet distance between embeddings distributions
    """
    try:
        from sentence_transformers import SentenceTransformer
        import scipy
    except ImportError:
        print("Please install sentence-transformers and scipy: pip install sentence-transformers scipy")
        return float('nan')
    
    # Load sentence embedding model
    model = SentenceTransformer(model_name)
    
    # Get embeddings
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    
    # Convert to numpy for calculation
    pred_embeddings = pred_embeddings.cpu().numpy()
    ref_embeddings = ref_embeddings.cpu().numpy()
    
    # Calculate mean and covariance for each distribution
    pred_mean = np.mean(pred_embeddings, axis=0)
    ref_mean = np.mean(ref_embeddings, axis=0)
    
    pred_cov = np.cov(pred_embeddings, rowvar=False)
    ref_cov = np.cov(ref_embeddings, rowvar=False)
    
    # Calculate Fréchet distance
    mean_diff = pred_mean - ref_mean
    mean_norm = np.sum(mean_diff ** 2)
    
    # Handle numerical stability
    covmean = scipy.linalg.sqrtm(pred_cov.dot(ref_cov))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Fréchet distance formula
    trace_term = np.trace(pred_cov + ref_cov - 2 * covmean)
    fd = mean_norm + trace_term
    
    return fd

def evaluate_model_on_dataset(
    model,
    processor,
    dataset,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        processor: The processor (tokenizer + image processor)
        dataset: The dataset to evaluate on
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    # Limit the number of samples if specified
    eval_dataset = dataset
    if max_samples is not None and max_samples < len(dataset):
        eval_dataset = dataset.select(range(max_samples))
    
    # Process samples in batches
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:min(i + batch_size, len(eval_dataset))]
        
        # Process images
        inputs = processor(
            images=batch["image"],
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
            )
        
        # Decode predictions
        batch_preds = processor.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
        # Get references
        batch_refs = batch["caption"]
        references.extend(batch_refs)
    
    # Calculate metrics
    metrics = compute_captioning_metrics(predictions, references)
    
    # Add example predictions for qualitative analysis
    metrics["examples"] = [
        {"image_id": i, "prediction": pred, "reference": ref}
        for i, (pred, ref) in enumerate(zip(predictions[:5], references[:5]))
    ]
    
    return metrics
