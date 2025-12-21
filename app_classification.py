import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# ====== Page config ======
st.set_page_config(
    page_title="Analyse des sentiments Amazon",
    page_icon="🧠",
    layout="wide"
)

# ====== CSS styling ======
st.markdown("""
<style>
/* Titres */
h1, h2, h3 { color: #1f4e79; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

/* Cartes métriques */
.stMetric {
    background-color: #eaf2f8 !important;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.15);
    color: #1f4e79 !important;
}

/* Sidebar */
.css-1d391kg { background-color: #f0f4f8; }

/* Boutons */
.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 8px;
    padding: 0.4em 1em;
    font-weight: bold;
}

/* DataFrame styling */
.stDataFrame div.row_widget.stDataFrame {
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Dashboard Interactif des Sentiments Amazon")

# ====== 1️⃣ Charger dataset ======
path_csv = st.text_input("Chemin du dataset nettoyé", "data/amazon_reviews_cleaned.csv")
try:
    df = pd.read_csv(path_csv)
except:
    st.error("Impossible de charger le fichier CSV.")
    st.stop()

# ====== 2️⃣ Détection des colonnes ======
text_col = next((col for col in df.columns if "review" in col.lower() and "text" in col.lower()), None)
label_col = 'label'
country_col = 'country' if 'country' in df.columns else None

if text_col is None or label_col not in df.columns:
    st.error("Colonnes texte ou label introuvables.")
    st.stop()

# ====== 3️⃣ Filtres interactifs ======
st.sidebar.header("Filtres interactifs")
# Filtre pays
if country_col:
    countries = df[country_col].unique().tolist()
    selected_countries = st.sidebar.multiselect("Sélectionner pays", countries, default=countries)
    df_filtered = df[df[country_col].isin(selected_countries)]
else:
    df_filtered = df.copy()

# Filtre sentiment
sentiments = [-1, 0, 1]
sentiment_map = {-1: "Négatif", 0: "Neutre", 1: "Positif"}
selected_sentiments = st.sidebar.multiselect(
    "Sélectionner sentiment", 
    sentiments, 
    default=sentiments, 
    format_func=lambda x: sentiment_map[x]
)
df_filtered = df_filtered[df_filtered[label_col].isin(selected_sentiments)]

st.write(f"Aperçu du dataset filtré : {len(df_filtered)} reviews")
st.dataframe(df_filtered.head())

# ====== 4️⃣ Séparation train/test ======
st.subheader("Séparation train/test")
test_size_percent = st.sidebar.slider("Pourcentage du dataset pour le test", 10, 50, 20, 5)
test_size = test_size_percent / 100

X = df_filtered[text_col]
y = df_filtered[label_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
st.write(f"Échantillons entraînement : {len(X_train)} | test : {len(X_test)} ({test_size_percent}%)")

# ====== 5️⃣ TF-IDF ======
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ====== 6️⃣ Choix des modèles ======
st.subheader("Choix des modèles")
model_choice = st.sidebar.multiselect(
    "Modèles à entraîner",
    ["Logistic Regression", "Naive Bayes"],
    default=["Logistic Regression", "Naive Bayes"]
)

models = {}
if "Logistic Regression" in model_choice:
    models["Logistic Regression"] = LogisticRegression(max_iter=500)
if "Naive Bayes" in model_choice:
    models["Naive Bayes"] = MultinomialNB()

if not models:
    st.warning("Veuillez sélectionner au moins un modèle.")
    st.stop()

# ====== 7️⃣ Entraînement et stockage ======
results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {'model': model, 'accuracy': accuracy_score(y_test, y_pred), 'y_pred': y_pred}

# ====== 8️⃣ Sélecteur modèle pour détails ======
st.subheader("Détails d’un modèle")
selected_model = st.selectbox("Choisir un modèle pour ses détails", list(results.keys()))
model_data = results[selected_model]

st.write(f"**Accuracy : {model_data['accuracy']:.4f}**")
report_df = pd.DataFrame(classification_report(y_test, model_data['y_pred'], output_dict=True)).transpose()
st.dataframe(report_df)

cm = confusion_matrix(y_test, model_data['y_pred'], labels=[1,0,-1])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,0,-1], yticklabels=[1,0,-1], ax=ax)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
ax.set_title(f"Matrice de confusion - {selected_model}")
st.pyplot(fig)

# ====== 9️⃣ Comparatif des modèles ======
st.subheader("Comparatif des modèles")
fig = px.bar(
    x=list(results.keys()),
    y=[v['accuracy'] for v in results.values()],
    text=[f"{v['accuracy']:.4f}" for v in results.values()],
    labels={"x": "Modèle", "y": "Accuracy"},
    color=list(results.keys()),
    color_discrete_map={
        "Logistic Regression": "#1f77b4",
        "Naive Bayes": "#ff7f0e"
    },
    height=400
)
st.plotly_chart(fig)
best_model = max(results, key=lambda x: results[x]['accuracy'])
st.success(f"Le meilleur modèle est **{best_model}** avec Accuracy = **{results[best_model]['accuracy']:.4f}**")

# ====== 🔟 Statistiques rapides ======
st.subheader("Statistiques rapides")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div style='background-color:#d4edda;padding:15px;border-radius:10px;text-align:center;'>Total reviews<br><b>{len(df_filtered)}</b></div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background-color:#cce5ff;padding:15px;border-radius:10px;text-align:center;'>Positives<br><b>{(df_filtered[label_col]==1).sum()}</b></div>", unsafe_allow_html=True)
col3.markdown(f"<div style='background-color:#fff3cd;padding:15px;border-radius:10px;text-align:center;'>Neutres<br><b>{(df_filtered[label_col]==0).sum()}</b></div>", unsafe_allow_html=True)
col4.markdown(f"<div style='background-color:#f8d7da;padding:15px;border-radius:10px;text-align:center;'>Négatives<br><b>{(df_filtered[label_col]==-1).sum()}</b></div>", unsafe_allow_html=True)

# ====== 11️⃣ Histogramme du nombre de mots ======
st.subheader("Distribution du nombre de mots par review")
df_filtered['review_length'] = df_filtered[text_col].apply(lambda x: len(str(x).split()))
fig = px.histogram(
    df_filtered,
    x='review_length',
    nbins=30,
    labels={'review_length':'Nombre de mots'},
    color_discrete_sequence=['#1f77b4']
)
st.plotly_chart(fig)

# ====== 12️⃣ Nuages de mots ======
st.subheader("Nuages de mots")
text_pos = " ".join(df_filtered[df_filtered[label_col]==1][text_col].dropna())
text_neg = " ".join(df_filtered[df_filtered[label_col]==-1][text_col].dropna())

if text_pos:
    st.markdown("**Reviews positives**")
    wc_pos = WordCloud(width=600, height=400, background_color='white', colormap='Blues', max_words=200).generate(text_pos)
    fig, ax = plt.subplots()
    ax.imshow(wc_pos, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

if text_neg:
    st.markdown("**Reviews négatives**")
    wc_neg = WordCloud(width=600, height=400, background_color='white', colormap='Reds', max_words=200).generate(text_neg)
    fig, ax = plt.subplots()
    ax.imshow(wc_neg, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
