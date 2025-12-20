import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(
    page_title="Analyse de Sentiments & Résumé Automatique - Amazon Reviews",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
h1 {color:#1f4e79; font-size:38px;}
h2 {color:#16365d;}
h3 {color:#2e75b6;}
.metric-box {background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0px 0px 10px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Choisissez la page",
    ["Introduction", "Statistiques & Graphes", "Dataset Nettoyé"]
)

st.sidebar.header("⚙️ Paramètres")
path_cleaned = st.sidebar.text_input("Chemin du dataset nettoyé", "data/amazon_reviews_cleaned.csv")

@st.cache_data
def load_data(p):
    try:
        df = pd.read_csv(p)
        return df
    except Exception as e:
        st.error(f"Impossible de charger le fichier CSV: {e}")
        return None

df = load_data(path_cleaned)
if df is None:
    st.stop()

review_text_col = next((col for col in df.columns if "review" in col.lower() and "text" in col.lower()), None)
review_title_col = next((col for col in df.columns if "title" in col.lower() or "summary" in col.lower()), None)
country_col = next((col for col in df.columns if "country" in col.lower() or "location" in col.lower()), None)

if 'label' not in df.columns and 'numeric_rating' in df.columns:
    def rating_to_sentiment(r):
        r = int(r)
        if r <= 2: return -1
        elif r == 3: return 0
        else: return 1
    df['label'] = df['numeric_rating'].apply(rating_to_sentiment)

# ================= PAGES =================
if page == "Introduction":
    st.title("🧠 Analyse de Sentiments et Résumé Automatique sur Amazon Reviews")
    st.markdown("""
    **Objectif du projet :**  
    Analyser les avis clients d’Amazon pour extraire des insights et préparer les données pour des modèles NLP.
    
    **Phases principales :**  

    1. **Prétraitement des données textuelles :**  
       - Nettoyage du texte déjà effectué dans le fichier `amazon_reviews_cleaned.csv`  
       - Suppression des caractères spéciaux, URLs, mentions, hashtags et emojis  
       - Tokenisation, suppression des stopwords et lemmatisation  
       - **Explication :** cette étape transforme le texte brut en un format standardisé et propre pour que les modèles NLP (TF-IDF, BERT, etc.) puissent mieux apprendre et détecter les sentiments.

    2. **Analyse de sentiments :**  
       - Entrée : avis clients  
       - Objectif : classifier en positif, neutre ou négatif  
       - Méthodes : TF-IDF + Logistic Regression / Naive Bayes, ou modèles BERT  

    3. **Résumé automatique :**  
       - Entrée : articles ou avis longs  
       - Objectif : générer un résumé concis pour une lecture rapide  
       - Méthodes : Transformers (BART, T5)  

    4. **Statistiques et visualisation :**  
       - Répartition des sentiments, distribution des ratings, top pays, nuages de mots, etc.  

    5. **Export du dataset nettoyé :**  
       - Permet l’utilisation pour d’autres modèles ou analyses NLP futures  

    **Dataset :** Amazon Reviews – contient avis clients, titres, notes, pays et autres métadonnées.
    """)
    st.subheader("Aperçu du dataset prétraité")
    st.write(f"Nombre total de reviews : **{len(df)}**")
    st.dataframe(df.head(40))

elif page == "Statistiques & Graphes":
    st.title("📊 Statistiques & Graphes")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", len(df))
    col2.metric("Reviews positives", (df['label'] == 1).sum())
    col3.metric("Reviews neutres", (df['label'] == 0).sum())
    col4.metric("Reviews négatives", (df['label'] == -1).sum())
    if country_col:
        st.metric("Nombre de pays", df[country_col].nunique())

    # Répartition des sentiments
    st.subheader("Répartition des sentiments")
    fig_sent, ax_sent = plt.subplots(figsize=(6,4))
    sns.countplot(x='label', data=df, palette='coolwarm', ax=ax_sent)
    ax_sent.set_xlabel("Sentiment (-1=Negatif, 0=Neutre, 1=Positif)")
    ax_sent.set_ylabel("Nombre de reviews")
    st.pyplot(fig_sent)

    # Distribution des ratings
    st.subheader("Distribution des ratings")
    fig_rate, ax_rate = plt.subplots(figsize=(6,4))
    sns.histplot(df['numeric_rating'], bins=5, kde=False, ax=ax_rate)
    ax_rate.set_xlabel("Rating")
    ax_rate.set_ylabel("Nombre de reviews")
    st.pyplot(fig_rate)

    # Top pays
    if country_col:
        st.subheader("Top 10 pays par nombre de reviews")
        top_countries = df[country_col].value_counts().head(10).reset_index()
        top_countries.columns = [country_col, 'Nombre de reviews']
        fig_country = px.bar(top_countries, 
                             x=country_col, y='Nombre de reviews', 
                             color='Nombre de reviews', color_continuous_scale='Blues',
                             width=600, height=400)
        st.plotly_chart(fig_country)

    # Nuages de mots
    st.subheader("Nuages de mots")
    text_pos = " ".join(df[df['label']==1][review_text_col].dropna())
    text_neg = " ".join(df[df['label']==-1][review_text_col].dropna())
    if text_pos:
        wc_pos = WordCloud(width=600, height=400, background_color='white').generate(text_pos)
        fig_wc_pos, ax_wc_pos = plt.subplots(figsize=(6,4))
        ax_wc_pos.imshow(wc_pos, interpolation='bilinear')
        ax_wc_pos.axis('off')
        ax_wc_pos.set_title("Reviews positives")
        st.pyplot(fig_wc_pos)
    if text_neg:
        wc_neg = WordCloud(width=600, height=600, background_color='white').generate(text_neg)
        fig_wc_neg, ax_wc_neg = plt.subplots(figsize=(6,4))
        ax_wc_neg.imshow(wc_neg, interpolation='bilinear')
        ax_wc_neg.axis('off')
        ax_wc_neg.set_title("Reviews négatives")
        st.pyplot(fig_wc_neg)

    # Longueur des reviews
    st.subheader("Distribution du nombre de mots par review")
    df['review_length'] = df[review_text_col].apply(lambda x: len(str(x).split()))
    fig_len, ax_len = plt.subplots(figsize=(6,4))
    sns.histplot(df['review_length'], bins=30, kde=True, ax=ax_len)
    ax_len.set_xlabel("Nombre de mots")
    ax_len.set_ylabel("Nombre de reviews")
    st.pyplot(fig_len)

    # Boxplot des sentiments par rating
    st.subheader("Sentiments selon les ratings")
    fig_box, ax_box = plt.subplots(figsize=(6,4))
    sns.boxplot(x='numeric_rating', y='label', data=df, ax=ax_box)
    ax_box.set_xlabel("Rating")
    ax_box.set_ylabel("Sentiment (-1=Negatif,0=Neutre,1=Positif)")
    st.pyplot(fig_box)

elif page == "Dataset Nettoyé":
    st.title("💾 Dataset Nettoyé")
    cols_show = [review_text_col, 'numeric_rating', 'label']
    if review_title_col:
        cols_show.insert(1, review_title_col)
    if country_col:
        cols_show.append(country_col)
    st.dataframe(df[cols_show].reset_index(drop=True))
    st.download_button(
        label="📥 Télécharger le dataset nettoyé",
        data=df[cols_show].to_csv(index=False).encode('utf-8'),
        file_name="amazon_reviews_cleaned.csv",
        mime="text/csv"
    )
