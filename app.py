import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from typing import List, Dict, Any
from transformers import pipeline
import io

from classification import run_classification

from summarization import summarize_text_tfidf

from dashboard import run_dashboard

st.set_page_config(
    page_title="Analyse de Sentiments Amazon",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
h1 {color:#1f4e79;}
h2 {color:#16365d;}
h3 {color:#2e75b6;}
</style>
""", unsafe_allow_html=True)

# ================================
# Fonctions Transformers avec cache Streamlit
# ================================

DEFAULT_HF_MODEL = "cmarkea/distilcamembert-base-sentiment"

@st.cache_resource(show_spinner="🔄 Chargement du modèle Transformers...")
def get_sentiment_pipeline(model_name: str = DEFAULT_HF_MODEL):
    """Charge et cache le pipeline HF pour analyse de sentiments"""
    try:
        return pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=-1,
            max_length=512,
            truncation=True
        )
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle '{model_name}': {str(e)}")
        return None

def normalize_label(raw_label: str, model_name: str) -> int:
    """Mappe les labels HF vers {-1, 0, 1}"""
    if raw_label is None:
        return 0
    
    lbl = str(raw_label).strip().lower()
    
    if any(k in lbl for k in ["neg", "-1", "negative"]):
        return -1
    if any(k in lbl for k in ["neu", "neutral", "0"]):
        return 0
    if any(k in lbl for k in ["pos", "+1", "positive"]):
        return 1
    
    for d in ["1", "2", "3", "4", "5"]:
        if d in lbl and "star" in lbl:
            val = int(d)
            if val <= 2:
                return -1
            elif val == 3:
                return 0
            else:
                return 1
    
    return 0

def hf_predict_text(text: str, model_name: str = DEFAULT_HF_MODEL) -> Dict[str, Any]:
    """Prédit le sentiment d'un texte unique"""
    try:
        nlp = get_sentiment_pipeline(model_name)
        if nlp is None:
            return {"error": "Pipeline indisponible"}
        
        text_truncated = text[:2000]
        out = nlp(text_truncated, truncation=True, max_length=512)[0]
        
        return {
            "label": out.get("label"),
            "score": float(out.get("score", 0.0)),
            "mapped": normalize_label(out.get("label"), model_name),
            "model": model_name
        }
    except Exception as e:
        return {"error": f"Erreur prédiction: {str(e)}"}

def hf_predict_batch(texts: List[str], model_name: str = DEFAULT_HF_MODEL) -> List[Dict[str, Any]]:
    """Prédit le sentiment de plusieurs textes"""
    try:
        nlp = get_sentiment_pipeline(model_name)
        if nlp is None:
            return []
        
        texts_truncated = [t[:2000] for t in texts]
        outputs = nlp(texts_truncated, truncation=True, max_length=512)
        
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
    except Exception as e:
        st.error(f"❌ Erreur batch: {str(e)}")
        return []

# ================================
# Navigation et chargement données
# ================================

st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Choisissez la page",
    [
        "Introduction",
        "Statistiques & Graphes",
        "Classification des Sentiments",
        "Résumé Automatique",
        "Analyse via Transformers",
        "Dashboard",
        "Dataset Nettoyé"
    ]
)

st.sidebar.header("⚙️ Paramètres")
path_cleaned = st.sidebar.text_input(
    "Chemin du dataset nettoyé",
    "data/amazon_reviews_cleaned.csv"
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(path_cleaned)
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

review_text_col = next(
    (c for c in df.columns if "review" in c.lower() and "text" in c.lower()),
    None
)

review_title_col = next(
    (c for c in df.columns if "title" in c.lower() or "summary" in c.lower()),
    None
)

country_col = next(
    (c for c in df.columns if "country" in c.lower()),
    None
)

if "label" not in df.columns and "numeric_rating" in df.columns:
    def rating_to_label(r):
        if r <= 2:
            return -1
        elif r == 3:
            return 0
        else:
            return 1
    df["label"] = df["numeric_rating"].apply(rating_to_label)

if page == "Introduction":
    st.title("🧠 Analyse de Sentiments sur les Avis Amazon")

    st.markdown("""
    **Objectif du projet :**  
    Analyser les avis clients Amazon afin de :
    - Nettoyer et structurer des données textuelles
    - Analyser les sentiments (positif / neutre / négatif)
    - Comparer des modèles de classification supervisée
    - Visualiser les résultats de manière interactive

    **Méthodologie :**
    - Prétraitement NLP (tokenisation, stopwords, lemmatisation)
    - Vectorisation TF-IDF
    - Modèles : Logistic Regression, Naive Bayes
    - Visualisation avec Streamlit
    """)

    st.subheader("Aperçu du dataset")
    st.write(f"Nombre total de reviews : **{len(df)}**")
    st.dataframe(df.head(30))

elif page == "Statistiques & Graphes":
    st.title("📊 Statistiques & Visualisations")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", len(df))
    col2.metric("Positives", (df["label"] == 1).sum())
    col3.metric("Neutres", (df["label"] == 0).sum())
    col4.metric("Négatives", (df["label"] == -1).sum())

    st.subheader("Répartition des sentiments")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax)
    ax.set_xlabel("Sentiment (-1, 0, 1)")
    ax.set_ylabel("Nombre")
    st.pyplot(fig)

    if "numeric_rating" in df.columns:
        st.subheader("Distribution des ratings")
        fig, ax = plt.subplots()
        sns.histplot(df["numeric_rating"], bins=5, ax=ax)
        st.pyplot(fig)

    if country_col:
        st.subheader("Top 10 pays")
        top = df[country_col].value_counts().head(10).reset_index()
        top.columns = ["Pays", "Nombre"]
        fig = px.bar(top, x="Pays", y="Nombre")
        st.plotly_chart(fig)

    st.subheader("Nuages de mots")
    text_pos = " ".join(df[df["label"] == 1][review_text_col].dropna())
    text_neg = " ".join(df[df["label"] == -1][review_text_col].dropna())

    col1, col2 = st.columns(2)
    if text_pos:
        wc = WordCloud(width=500, height=400, background_color="white").generate(text_pos)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title("Positifs")
        col1.pyplot(fig)

    if text_neg:
        wc = WordCloud(width=500, height=400, background_color="white").generate(text_neg)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title("Négatifs")
        col2.pyplot(fig)

elif page == "Classification des Sentiments":
    st.title("🤖 Classification des Sentiments")

    st.subheader("⚙️ Paramètres")

    test_size = st.slider(
        "Taille du jeu de test (%)",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    ) / 100

    models = st.multiselect(
        "Choisir les modèles",
        ["Logistic Regression", "Naive Bayes"],
        default=["Logistic Regression", "Naive Bayes"]
    )

    if st.button("🚀 Lancer la classification"):
        with st.spinner("Entraînement et évaluation des modèles..."):
            results = run_classification(
                df,
                review_text_col,
                "label",
                test_size,
                models
            )

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if not valid_results:
            st.error("❌ Aucun modèle n'a pu être entraîné.")
            for model_name, result in results.items():
                if "error" in result:
                    st.error(f"{model_name} : {result['error']}")
            st.stop()

        # =========================
        # Infos globales Train/Test
        # =========================
        st.subheader("📌 Répartition des données")

        example_model = next(iter(valid_results.values()))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train (%)", f"{example_model['train_percent']} %")
        col2.metric("Test (%)", f"{example_model['test_percent']} %")
        col3.metric("Train size", example_model["train_size"])
        col4.metric("Test size", example_model["test_size"])

        # =========================
        # Comparaison des modèles
        # =========================
        st.subheader("🏁 Comparaison des modèles")

        best_model = max(valid_results.items(), key=lambda x: x[1]["accuracy"])
        st.success(
            f"🏆 **Meilleur modèle : {best_model[0]}** "
            f"(Accuracy = {best_model[1]['accuracy']:.4f})"
        )

        fig = px.bar(
            x=list(valid_results.keys()),
            y=[v["accuracy"] for v in valid_results.values()],
            text=[f"{v['accuracy']:.3f}" for v in valid_results.values()],
            labels={"x": "Modèle", "y": "Accuracy"},
            title="Comparaison des accuracies"
        )
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Détails par modèle
        # =========================
        st.subheader("📋 Détails des modèles")

        for model_name, model in valid_results.items():
            with st.expander(f"📊 {model_name}"):

                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{model['accuracy']:.4f}")
                col2.metric("Temps d'entraînement (s)", model["training_time"])

                st.markdown("**📑 Rapport de classification**")
                st.dataframe(pd.DataFrame(model["report"]).transpose())

                st.markdown("**🔍 Matrice de confusion**")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    model["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=["Négatif", "Neutre", "Positif"],
                    yticklabels=["Négatif", "Neutre", "Positif"]
                )
                ax.set_xlabel("Prédiction")
                ax.set_ylabel("Réel")
                ax.set_title(f"Matrice de confusion – {model_name}")
                st.pyplot(fig)

      
        st.success(
            "💾 Le meilleur modèle a été sauvegardé avec le vectoriseur TF-IDF "
            "dans le fichier **sentiment_model.pkl**"
        )

elif page == "Dataset Nettoyé":
    st.title("💾 Dataset Nettoyé")

    cols = [review_text_col, "numeric_rating", "label"]
    if review_title_col:
        cols.insert(1, review_title_col)
    if country_col:
        cols.append(country_col)

    st.dataframe(df[cols].reset_index(drop=True))

    st.download_button(
        "📥 Télécharger le dataset",
        data=df[cols].to_csv(index=False).encode("utf-8"),
        file_name="amazon_reviews_cleaned.csv",
        mime="text/csv"
    )

elif page == "Analyse via Transformers":
    st.title("🤖 Analyse de Sentiments via Transformers")
    
    st.markdown("""
    Cette page utilise des modèles **Transformers pré-entraînés** de Hugging Face 
    pour l'analyse de sentiments. Ces modèles utilisent des architectures avancées 
    (BERT, CamemBERT, XLM-RoBERTa) et offrent généralement de meilleures performances 
    que les méthodes TF-IDF classiques.
    """)

    if review_text_col is None:
        st.error("❌ Aucune colonne de texte détectée dans le dataset.")
        st.stop()

    st.subheader("⚙️ Configuration du modèle")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_name = st.selectbox(
            "Sélectionnez le modèle Hugging Face",
            options=[
                "cmarkea/distilcamembert-base-sentiment",
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                "nlptown/bert-base-multilingual-uncased-sentiment",
            ],
            index=0,
            help="CamemBERT pour français, XLM-RoBERTa pour multilingue"
        )
    
    with col2:
        st.info(f"""
        **Modèle actuel:**  
        `{model_name.split('/')[-1][:25]}...`
        """)

    # =========================
    # Prédiction texte unique
    # =========================
    st.subheader("📝 Test sur un texte unique")
    
    with st.expander("✍️ Entrez votre texte", expanded=True):
        user_text = st.text_area(
            "Texte à analyser",
            value="Ce produit est absolument fantastique ! Je le recommande vivement.",
            height=120,
            help="Entrez n'importe quel avis ou commentaire"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            predict_btn = st.button("🔮 Analyser", type="primary", use_container_width=True)
        
        if predict_btn and user_text.strip():
            try:
                with st.spinner("🔄 Analyse en cours..."):
                    res = hf_predict_text(user_text, model_name)
                
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                else:
                    st.success("✅ Analyse terminée !")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_emoji = {1: "😊", 0: "😐", -1: "😞"}
                    sentiment_name = {1: "Positif", 0: "Neutre", -1: "Négatif"}
                    sentiment_color = {1: "normal", 0: "off", -1: "inverse"}
                    
                    mapped = res["mapped"]
                    
                    col1.metric(
                        "Sentiment",
                        f"{sentiment_emoji[mapped]} {sentiment_name[mapped]}",
                        delta=None
                    )
                    col2.metric(
                        "Confiance",
                        f"{res['score']:.1%}",
                        delta=None
                    )
                    col3.metric(
                        "Label brut",
                        res["label"],
                        delta=None
                    )
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                st.exception(e)

    # =========================
    # Évaluation sur dataset
    # =========================
    st.subheader("📊 Évaluation sur le dataset")
    
    st.markdown("""
    Testez le modèle sur un échantillon de votre dataset pour évaluer ses performances 
    et comparer avec les modèles ML classiques.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sample_size = st.slider(
            "Taille de l'échantillon",
            min_value=50,
            max_value=min(2000, len(df)),
            value=min(500, len(df)),
            step=50,
            help="Plus l'échantillon est grand, plus l'évaluation est précise (mais plus lente)"
        )
    
    with col2:
        st.metric("Dataset total", f"{len(df):,} avis")
    
    eval_btn = st.button("🚀 Lancer l'évaluation", type="primary", use_container_width=True)
    
    # Initialiser les clés du session_state si elles n'existent pas
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    
    # Exécuter l'évaluation et stocker les résultats dans session_state
    if eval_btn:
        try:
            with st.spinner(f"🔄 Inférence sur {sample_size} avis en cours..."):
                # Échantillonnage stratifié si possible
                if "label" in df.columns and df["label"].nunique() > 1:
                    df_sample = df.groupby("label", group_keys=False).apply(
                        lambda x: x.sample(min(len(x), sample_size // df["label"].nunique()), random_state=42)
                    ).sample(frac=1, random_state=42).head(sample_size)
                else:
                    df_sample = df.sample(n=sample_size, random_state=42)
                
                # Réinitialiser l'index de df_sample pour garantir la cohérence
                df_sample = df_sample.reset_index(drop=True)
                
                # Ajouter un identifiant unique pour traçabilité
                df_sample["eval_id"] = range(len(df_sample))
                
                texts = df_sample[review_text_col].astype(str).tolist()
                results = hf_predict_batch(texts, model_name)
                
                if not results:
                    st.error("❌ Aucun résultat. Vérifiez le modèle et la connexion internet.")
                    st.stop()
                
                df_pred = pd.DataFrame(results).reset_index(drop=True)
                df_pred["eval_id"] = range(len(df_pred))
                
                # Créer le DataFrame merged pour garantir la cohérence absolue
                # Les deux DataFrames ont maintenant des index alignés (0, 1, 2, ...)
                merged = pd.concat([
                    df_sample[["eval_id", review_text_col, "label"]],
                    df_pred[["label", "score", "mapped"]].add_prefix("hf_")
                ], axis=1)
                
                # Vérification de cohérence
                assert len(df_sample) == len(df_pred) == len(merged), "Désalignement détecté !"
                
                # Stocker les résultats dans session_state pour persistance
                st.session_state.eval_results = {
                    "df_sample": df_sample,
                    "df_pred": df_pred,
                    "sample_size": sample_size,
                    "model_name": model_name,
                    "review_text_col": review_text_col,
                    "merged": merged
                }
                
                st.success(f"✅ Évaluation terminée ! {len(merged)} prédictions figées dans l'état de l'application.")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'évaluation : {str(e)}")
            st.session_state.eval_results = None
    
    # Afficher les résultats uniquement si une évaluation a été effectuée
    if st.session_state.eval_results is not None:
        # Récupérer les résultats stockés
        df_sample = st.session_state.eval_results["df_sample"]
        df_pred = st.session_state.eval_results["df_pred"]
        sample_size = st.session_state.eval_results["sample_size"]
        stored_model = st.session_state.eval_results["model_name"]
        review_text_col = st.session_state.eval_results["review_text_col"]
        merged = st.session_state.eval_results["merged"]
        
        # Afficher un bandeau de confirmation de cohérence
        st.info(f"🔒 **Résultats figés** : {sample_size} prédictions synchronisées entre affichage et export")
        
        # Avertir si le modèle a changé depuis l'évaluation
        if stored_model != model_name:
            st.warning(f"⚠️ Les résultats affichés proviennent du modèle **{stored_model}**. Relancez l'évaluation pour utiliser **{model_name}**.")
        
        try:
            # =========================
            # Distribution des prédictions
            # =========================
            st.markdown("---")
            st.subheader("📈 Distribution des prédictions")
            
            dist = df_pred["mapped"].value_counts().sort_index()
            
            fig = px.bar(
                x=["Négatif (-1)", "Neutre (0)", "Positif (+1)"],
                y=[dist.get(-1, 0), dist.get(0, 0), dist.get(1, 0)],
                labels={"x": "Sentiment", "y": "Nombre d'avis"},
                title=f"Répartition des sentiments prédits (n={sample_size})",
                color=["Négatif (-1)", "Neutre (0)", "Positif (+1)"],
                color_discrete_map={
                    "Négatif (-1)": "#ef4444",
                    "Neutre (0)": "#f59e0b",
                    "Positif (+1)": "#10b981"
                }
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("😞 Négatifs", dist.get(-1, 0), f"{100*dist.get(-1, 0)/sample_size:.1f}%")
            col2.metric("😐 Neutres", dist.get(0, 0), f"{100*dist.get(0, 0)/sample_size:.1f}%")
            col3.metric("😊 Positifs", dist.get(1, 0), f"{100*dist.get(1, 0)/sample_size:.1f}%")
            
            # =========================
            # Métriques de performance
            # =========================
            if "label" in df_sample.columns:
                st.markdown("---")
                st.subheader("🎯 Performance du modèle")
                
                y_true = df_sample["label"].tolist()
                y_pred = df_pred["mapped"].tolist()
                
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                acc = accuracy_score(y_true, y_pred)
                
                # Métriques principales
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 Accuracy", f"{acc:.2%}")
                
                report = classification_report(
                    y_true, y_pred,
                    labels=[-1, 0, 1],
                    target_names=["Négatif", "Neutre", "Positif"],
                    zero_division=0,
                    output_dict=True
                )
                
                col2.metric("📊 F1-Score (macro)", f"{report['macro avg']['f1-score']:.2%}")
                col3.metric("📊 Precision (macro)", f"{report['macro avg']['precision']:.2%}")
                
                # Rapport détaillé
                with st.expander("📋 Rapport de classification détaillé"):
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(
                        df_report.style.format("{:.3f}"),
                        use_container_width=True
                    )
                
                # Matrice de confusion
                st.markdown("**🔍 Matrice de confusion**")
                
                cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
                
                fig_cm, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=["Négatif", "Neutre", "Positif"],
                    yticklabels=["Négatif", "Neutre", "Positif"],
                    cbar_kws={"label": "Nombre d'avis"}
                )
                ax.set_xlabel("Prédiction", fontsize=12)
                ax.set_ylabel("Réel", fontsize=12)
                ax.set_title(f"Matrice de confusion – {model_name.split('/')[-1]}", fontsize=14, pad=20)
                st.pyplot(fig_cm)
                
                # Analyse des erreurs
                with st.expander("🔎 Exemples d'erreurs de classification"):
                    df_errors = df_sample.copy()
                    df_errors["prediction"] = y_pred
                    df_errors["correct"] = df_errors["label"] == df_errors["prediction"]
                    df_errors_only = df_errors[~df_errors["correct"]].head(10)
                    
                    if len(df_errors_only) > 0:
                        for idx, row in df_errors_only.iterrows():
                            sentiment_name = {1: "Positif", 0: "Neutre", -1: "Négatif"}
                            st.markdown(f"""
                            **Texte:** {row[review_text_col][:200]}...  
                            **Réel:** {sentiment_name[row['label']]} | **Prédit:** {sentiment_name[row['prediction']]}
                            """)
                            st.markdown("---")
                    else:
                        st.success("✅ Aucune erreur dans les 10 premiers exemples !")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de l'affichage: {str(e)}")
            st.exception(e)

elif page == "Résumé Automatique":
    st.title("📝 Résumé Automatique (TF-IDF)")

    st.markdown("""
    Cette section utilise l'approche **Extractive TF-IDF**. 
    Elle identifie et extrait les segments les plus riches en mots-clés 
    pour générer un résumé fidèle au texte original.
    """)

    mode = st.radio(
        "Source du texte",
        ["Saisir un texte manuel", "Sélectionner une review du dataset"]
    )

    n_sent = st.slider("Nombre de segments à extraire", 1, 10, 3)

    if mode == "Saisir un texte manuel":
        user_text = st.text_area("Collez une review ici :", height=200)
        if st.button("✨ Résumer"):
            if user_text:
                summary = summarize_text_tfidf(user_text, n_sent)
                st.success("Résumé :")
                st.write(summary)
            else:
                st.warning("Veuillez entrer du texte.")

    else:
        if review_text_col:
            # On prend les 100 premières pour la fluidité du selectbox
            options = df[review_text_col].dropna().unique()[:100]
            selected_review = st.selectbox("Choisissez une review :", options)
            
            if st.button("✨ Résumer cette review"):
                summary = summarize_text_tfidf(selected_review, n_sent)
                st.info("📌 Texte original :")
                st.write(selected_review)
                st.success("✅ Résumé :")
                st.write(summary)

elif page == "Dashboard":
    run_dashboard(df, review_text_col, country_col)