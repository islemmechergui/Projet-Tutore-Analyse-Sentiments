import streamlit as st
from typing import Dict, Any, List
import time
from datetime import datetime


def get_sentiment_emoji(mapped_label: int) -> str:
    """Retourne l'emoji correspondant au sentiment"""
    sentiment_emojis = {
        1: "😊",
        0: "😐",
        -1: "😞"
    }
    return sentiment_emojis.get(mapped_label, "❓")


def get_sentiment_name(mapped_label: int) -> str:
    """Retourne le nom du sentiment"""
    sentiment_names = {
        1: "Positif",
        0: "Neutre",
        -1: "Négatif"
    }
    return sentiment_names.get(mapped_label, "Inconnu")


def get_sentiment_color(mapped_label: int) -> str:
    """Retourne la couleur associée au sentiment"""
    colors = {
        1: "#10b981",  # vert
        0: "#f59e0b",  # orange
        -1: "#ef4444"  # rouge
    }
    return colors.get(mapped_label, "#gray")


def get_confidence_interpretation(score: float) -> str:
    """Interprète le score de confiance"""
    if score >= 0.9:
        return "très élevée"
    elif score >= 0.7:
        return "élevée"
    elif score >= 0.5:
        return "modérée"
    else:
        return "faible"


def format_analysis_response(result: Dict[str, Any]) -> str:
    """Formate la réponse d'analyse en texte naturel"""
    if "error" in result:
        return f"❌ Désolé, une erreur s'est produite : {result['error']}"
    
    mapped = result.get("mapped", 0)
    score = result.get("score", 0.0)
    
    sentiment = get_sentiment_name(mapped)
    emoji = get_sentiment_emoji(mapped)
    confidence = get_confidence_interpretation(score)
    
    response = f"{emoji} **Sentiment détecté : {sentiment}**\n\n"
    response += f"Confiance : {score:.1%} ({confidence})\n\n"
    
    # Ajout de contexte selon le sentiment
    if mapped == 1:
        response += "✨ Ce texte exprime une opinion positive. "
        if score >= 0.8:
            response += "Le modèle est très confiant dans cette classification."
        else:
            response += "Cependant, il pourrait contenir quelques nuances."
    elif mapped == -1:
        response += "⚠️ Ce texte exprime une opinion négative. "
        if score >= 0.8:
            response += "Le modèle détecte clairement des éléments négatifs."
        else:
            response += "Il pourrait y avoir des aspects mitigés."
    else:
        response += "ℹ️ Ce texte semble neutre ou équilibré entre aspects positifs et négatifs."
    
    return response


def generate_suggestions(mapped_label: int) -> List[str]:
    """Génère des suggestions de textes à tester selon le dernier sentiment"""
    suggestions = {
        1: [
            "Ce produit est décevant, je ne le recommande pas.",
            "Service client catastrophique, à éviter absolument.",
            "Rapport qualité-prix correct mais rien d'exceptionnel."
        ],
        -1: [
            "Excellent produit, très satisfait de mon achat !",
            "Livraison rapide et article conforme à la description.",
            "C'est acceptable mais il y a des points à améliorer."
        ],
        0: [
            "Produit absolument parfait, je recommande vivement !",
            "Très mauvaise expérience, totalement déçu.",
            "Conforme à mes attentes, ni plus ni moins."
        ]
    }
    return suggestions.get(mapped_label, [])


def initialize_chat_history():
    """Initialise l'historique du chat dans session_state"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "👋 Bonjour ! Je suis votre assistant d'analyse de sentiments.\n\n"
                          "Envoyez-moi n'importe quel texte (avis, commentaire, message) "
                          "et je vous dirai s'il est **positif**, **négatif** ou **neutre** !\n\n"
                          "💡 Vous pouvez aussi tester les exemples suggérés ci-dessous.",
                "timestamp": datetime.now()
            }
        ]


def add_message(role: str, content: str):
    """Ajoute un message à l'historique"""
    st.session_state.chat_messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })


def run_chatbot(predict_function, model_name: str):
    """
    Exécute l'interface du chatbot
    
    Args:
        predict_function: Fonction de prédiction (hf_predict_text)
        model_name: Nom du modèle Hugging Face utilisé
    """
    
    st.title("💬 Chatbot d'Analyse de Sentiments")
    
    
    
    # Initialisation
    initialize_chat_history()
    
    # Configuration dans la sidebar
    with st.sidebar:
        st.header("⚙️ Configuration du chatbot")
        
        show_confidence = st.checkbox("Afficher les scores de confiance", value=True)
        show_suggestions = st.checkbox("Afficher les suggestions", value=True)
        
        st.markdown("---")
        
        if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
            st.session_state.chat_messages = []
            initialize_chat_history()
            st.rerun()
        
        st.markdown("---")
        st.info(f"**Modèle actif:**\n{model_name}")
        st.metric("Messages", len(st.session_state.chat_messages))
    
    # Zone d'affichage des messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
    
    # Suggestions interactives
    if show_suggestions and len(st.session_state.chat_messages) > 1:
        last_analysis = None
        for msg in reversed(st.session_state.chat_messages):
            if msg["role"] == "assistant" and "Sentiment détecté" in msg["content"]:
                # Extraire le sentiment de la dernière analyse
                if "Positif" in msg["content"]:
                    last_analysis = 1
                elif "Négatif" in msg["content"]:
                    last_analysis = -1
                else:
                    last_analysis = 0
                break
        
        if last_analysis is not None:
            suggestions = generate_suggestions(last_analysis)
            st.markdown("**💡 Essayez aussi ces exemples :**")
            cols = st.columns(len(suggestions))
            for idx, (col, suggestion) in enumerate(zip(cols, suggestions)):
                if col.button(f"Exemple {idx+1}", key=f"sugg_{idx}", use_container_width=True):
                    # Ajouter comme message utilisateur
                    add_message("user", suggestion)
                    
                    # Analyser
                    with st.spinner("🔄 Analyse en cours..."):
                        result = predict_function(suggestion, model_name)
                        response = format_analysis_response(result)
                        add_message("assistant", response)
                    
                    st.rerun()
            
            # Afficher les textes des suggestions en petit
            for idx, suggestion in enumerate(suggestions):
                st.caption(f"**Ex. {idx+1}:** {suggestion[:60]}...")
    
    # Zone de saisie
    user_input = st.chat_input("Entrez votre texte à analyser...")
    
    if user_input:
        # Afficher le message de l'utilisateur
        add_message("user", user_input)
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        
        # Analyser et répondre
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔄 Analyse en cours..."):
                time.sleep(0.5)  # Petit délai pour l'effet de "réflexion"
                result = predict_function(user_input, model_name)
                response = format_analysis_response(result)
                
                st.markdown(response)
                
                # Afficher les détails techniques si demandé
                if show_confidence and "error" not in result:
                    with st.expander("📊 Détails techniques"):
                        col1, col2 = st.columns(2)
                        col1.metric("Label brut", result.get("label", "N/A"))
                        col2.metric("Score", f"{result.get('score', 0):.4f}")
                        col1.metric("Mapping", result.get("mapped", 0))
                        col2.metric("Modèle", result.get("model", "N/A").split('/')[-1])
        
        add_message("assistant", response)
        st.rerun()
    
    # Exemples de démarrage rapide
    if len(st.session_state.chat_messages) == 1:
        st.markdown("---")
        st.markdown("### 🚀 Exemples pour commencer")
        
        quick_examples = [
            "Ce produit dépasse toutes mes attentes, vraiment excellent !",
            "Déçu par la qualité, je ne recommande pas du tout.",
            "Produit correct, conforme à la description."
        ]
        
        cols = st.columns(len(quick_examples))
        for idx, (col, example) in enumerate(zip(cols, quick_examples)):
            if col.button(f"Tester", key=f"quick_{idx}", use_container_width=True):
                add_message("user", example)
                with st.spinner("🔄 Analyse en cours..."):
                    result = predict_function(example, model_name)
                    response = format_analysis_response(result)
                    add_message("assistant", response)
                st.rerun()
        
        for idx, example in enumerate(quick_examples):
            cols[idx].caption(f"{example[:40]}...")