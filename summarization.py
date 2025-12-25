import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_text_tfidf(text, n_sentences=3):
    if not text or len(text.strip()) < 20:
        return text

    # Découpage par ponctuation ou par segments de 15 mots si pas de points
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    if len(sentences) <= 1:
        words = text.split()
        if len(words) > 15:
            sentences = [" ".join(words[i:i+15]) for i in range(0, len(words), 15)]
        else:
            return text

    try:
        # On ignore les mots vides anglais pour le calcul des scores
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Score de chaque "phrase" = somme de ses scores TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Sélection des N meilleures phrases
        n_sentences = min(n_sentences, len(sentences))
        top_indices = np.argsort(sentence_scores)[-n_sentences:]
        top_indices.sort()  # Garder l'ordre du texte original
        
        return " ".join([sentences[i] for i in top_indices])
    except:
        return text