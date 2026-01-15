"""
Module d'analyse de sentiments via Transformers (Hugging Face)
Fournit des pipelines pré-entraînés pour l'inférence de sentiments
"""

from typing import List, Dict, Any, Optional, Literal
from transformers import pipeline
from functools import lru_cache
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "cmarkea/distilcamembert-base-sentiment"

# Type hints pour les sentiments
SentimentLabel = Literal[-1, 0, 1]


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
        logger.info(f"Chargement du modèle: {model_name}")
        return pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU; set 0 for CUDA if available
            truncation=True  # Déjà défini ici pour tous les appels
        )
    except Exception as e:
        logger.error(f"Erreur de chargement du modèle '{model_name}': {e}")
        return None


def normalize_label(raw_label: str, model_name: str) -> SentimentLabel:
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
    logger.warning(f"Label inconnu '{raw_label}' pour {model_name}, mapping à neutral")
    return 0


def hf_predict_text(
    text: str, 
    model_name: str = DEFAULT_HF_MODEL,
    return_all_scores: bool = False
) -> Dict[str, Any]:
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text to analyze
        model_name: Hugging Face model identifier
        return_all_scores: If True, return scores for all labels
        
    Returns:
        Dictionary with label, score, mapped label, and model name
    """
    if not text or not text.strip():
        return {
            "error": "Texte vide",
            "label": None,
            "score": 0.0,
            "mapped": 0,
            "model": model_name
        }
    
    nlp = get_sentiment_pipeline(model_name)
    if nlp is None:
        return {"error": "Pipeline indisponible"}
    
    try:
        out = nlp(text, return_all_scores=return_all_scores)[0]
        
        result = {
            "label": out.get("label"),
            "score": float(out.get("score", 0.0)),
            "mapped": normalize_label(out.get("label"), model_name),
            "model": model_name
        }
        
        if return_all_scores and isinstance(out, list):
            result["all_scores"] = out
            
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return {"error": str(e)}


def hf_predict_batch(
    texts: List[str], 
    model_name: str = DEFAULT_HF_MODEL,
    batch_size: int = 8,
    skip_empty: bool = True
) -> List[Dict[str, Any]]:
    """
    Predict sentiment for multiple texts
    
    Args:
        texts: List of input texts to analyze
        model_name: Hugging Face model identifier
        batch_size: Number of texts to process at once
        skip_empty: If True, skip empty texts
        
    Returns:
        List of dictionaries with predictions for each text
    """
    if not texts:
        return []
    
    # Filtrage optionnel des textes vides
    if skip_empty:
        texts_filtered = [t for t in texts if t and t.strip()]
        if len(texts_filtered) != len(texts):
            logger.warning(f"{len(texts) - len(texts_filtered)} textes vides ignorés")
    else:
        texts_filtered = texts
    
    if not texts_filtered:
        return []
    
    nlp = get_sentiment_pipeline(model_name)
    if nlp is None:
        return [{"error": "Pipeline indisponible"} for _ in texts_filtered]
    
    try:
        outputs = nlp(texts_filtered, truncation=True, batch_size=batch_size)
        results = []
        for t, r in zip(texts_filtered, outputs):
            results.append({
                "text": t[:100] + "..." if len(t) > 100 else t,  # Tronquer pour log
                "label": r.get("label"),
                "score": float(r.get("score", 0.0)),
                "mapped": normalize_label(r.get("label"), model_name),
                "model": model_name
            })
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}")
        return [{"error": str(e)} for _ in texts_filtered]


def get_model_info(model_name: str = DEFAULT_HF_MODEL) -> Optional[Dict[str, Any]]:
    """
    Get information about a loaded model
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        Dictionary with model information or None
    """
    nlp = get_sentiment_pipeline(model_name)
    if nlp is None:
        return None
    
    return {
        "model_name": model_name,
        "task": nlp.task,
        "device": str(nlp.device),
        "framework": nlp.framework
    }