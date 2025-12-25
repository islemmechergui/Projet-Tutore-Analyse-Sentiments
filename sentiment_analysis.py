"""
Module d'analyse de sentiments via Transformers (Hugging Face)
Fournit des pipelines pré-entraînés pour l'inférence de sentiments
"""

from typing import List, Dict, Any
from transformers import pipeline
from functools import lru_cache


DEFAULT_HF_MODEL = "cmarkea/distilcamembert-base-sentiment"


@lru_cache(maxsize=3)
def get_sentiment_pipeline(model_name: str = DEFAULT_HF_MODEL):
    """
    Load and cache HF sentiment analysis pipeline
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        pipeline object or None if error
    """
    try:
        return pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU; set 0 for CUDA if available
        )
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle '{model_name}': {e}")
        return None


def normalize_label(raw_label: str, model_name: str) -> int:
    """
    Map HF labels to {-1, 0, 1}
    Handles POSITIVE/NEGATIVE/NEUTRAL and 1-5 stars formats
    
    Args:
        raw_label: Label from HF model (e.g., "POSITIVE", "1 star")
        model_name: Model identifier for context
        
    Returns:
        -1 (negative), 0 (neutral), or 1 (positive)
    """
    if raw_label is None:
        return 0
    
    lbl = str(raw_label).strip().lower()

    # Generic mapping
    if any(k in lbl for k in ["neg", "-1", "negative"]):
        return -1
    if any(k in lbl for k in ["neu", "neutral", "0"]):
        return 0
    if any(k in lbl for k in ["pos", "+1", "positive"]):
        return 1

    # Stars format: "1 star", "2 stars", etc.
    for d in ["1", "2", "3", "4", "5"]:
        if d in lbl and "star" in lbl:
            val = int(d)
            if val <= 2:
                return -1
            elif val == 3:
                return 0
            else:
                return 1

    # Fallback: neutral
    return 0


def hf_predict_text(text: str, model_name: str = DEFAULT_HF_MODEL) -> Dict[str, Any]:
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text to analyze
        model_name: Hugging Face model identifier
        
    Returns:
        Dictionary with label, score, mapped label, and model name
    """
    nlp = get_sentiment_pipeline(model_name)
    if nlp is None:
        return {"error": "Pipeline indisponible"}
    
    out = nlp(text)[0]
    return {
        "label": out.get("label"),
        "score": float(out.get("score", 0.0)),
        "mapped": normalize_label(out.get("label"), model_name),
        "model": model_name
    }


def hf_predict_batch(texts: List[str], model_name: str = DEFAULT_HF_MODEL) -> List[Dict[str, Any]]:
    """
    Predict sentiment for multiple texts
    
    Args:
        texts: List of input texts to analyze
        model_name: Hugging Face model identifier
        
    Returns:
        List of dictionaries with predictions for each text
    """
    nlp = get_sentiment_pipeline(model_name)
    if nlp is None:
        return []
    
    outputs = nlp(texts, truncation=True)
    results = []
    for t, r in zip(texts, outputs):
        results.append({
            "text": t,
            "label": r.get("label"),
            "score": float(r.get("score", 0.0)),
            "mapped": normalize_label(r.get("label"), model_name),
            "model": model_name
        })
    return results
