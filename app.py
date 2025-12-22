import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

from classification import run_classification

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

st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Choisissez la page",
    [
        "Introduction",
        "Statistiques & Graphes",
        "Classification des Sentiments",
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