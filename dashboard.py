import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

def run_dashboard(df, review_text_col, country_col):
    # CSS personnalisé pour transformer les colonnes en cartes
    st.markdown("""
        <style>
        /* 1. Titre principal du Dashboard */
        h1 {
            color: #1f4e79;
            font-family: 'Arial', sans-serif;
            text-align: center;
            padding-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        /* 2. Sous-titres des sections (Graphes) */
        h3 {
            color: #2e75b6;
            background: linear-gradient(90deg, #1f4e79 0%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-top: 15px;
            border-bottom: 2px solid #f0f2f6;
        }

        /* 3. Titres à l'intérieur des cartes KPI */
        [data-testid="stMetricLabel"] p {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #1f4e79 !important;
            text-transform: uppercase;
            text-align: center;
            opacity: 0.9;
        }

        /* Conteneur principal de la carte */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            border: 1px solid #e6e9ef;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


    # --- 1. Style CSS pour les conteneurs (Background & Ombres) ---
    st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        .stPlotlyChart {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title(" Dashboard Analytique ")

    if 'word_count' not in df.columns:
        df['word_count'] = df[review_text_col].astype(str).apply(lambda x: len(x.split()))
    
    tab1, tab2, tab3 = st.tabs(["📈 KPIs & Distribution", "🔍 Analyse Sémantique", "🌍 Géographie"])

    with tab1:
        avg_rating = df['numeric_rating'].mean() if 'numeric_rating' in df.columns else 0
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Reviews", len(df))
        col2.metric("Reviews Positifs", (df["label"] == 1).sum())
        col3.metric("Reviews Neutres", (df["label"] == 0).sum())
        col4.metric("Reviews Négatifs", (df["label"] == -1).sum())
        col5.metric("Rating Moyen", f"{avg_rating:.2f} / 5")

        st.subheader("Distribution des Sentiments par Rating")
        fig_violin = px.violin(
            df, y="numeric_rating", x="label", color="label", 
            box=True, points="all", title="Dispersion des Ratings",
            color_discrete_map={-1: "#EF553B", 0: "#FECB52", 1: "#00CC96"},
            template="plotly_white" # Thème épuré
        )
        # Personnalisation du fond
        fig_violin.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_violin, use_container_width=True)

    with tab2:
        st.subheader("Analyse Sémantique Approfondie")
        
        # Fonction améliorée avec Bigrammes et Stopwords
        def get_top_phrases(sentiment_label, top_n=10):
            text = " ".join(df[df["label"] == sentiment_label][review_text_col].dropna().astype(str).str.lower())
            custom_stopwords = {'amazon', 'product', 'item', 'would', 'customer', 'really', 'this', 'that'}
            
            words = [w for w in text.split() if len(w) > 3 and w not in custom_stopwords]
            # Création de bigrammes (ex: "excellent service")
            bigrams = [" ".join(pair) for pair in zip(words, words[1:])]
            
            return pd.DataFrame(Counter(bigrams).most_common(top_n), columns=['Phrase', 'Fréquence'])

        df_pos = get_top_phrases(1)
        df_neg = get_top_phrases(-1)

        col_left, col_right = st.columns(2)

        with col_left:
            st.write("**✅ Top 10 Expressions Positives**")
            fig_pos = px.bar(df_pos, x='Fréquence', y='Phrase', orientation='h',
                             color='Fréquence', color_continuous_scale='Greens',
                             template="plotly_white")
            fig_pos.update_layout(
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_pos, use_container_width=True)

        with col_right:
            st.write("**❌ Top 10 Expressions Négatives**")
            fig_neg = px.bar(df_neg, x='Fréquence', y='Phrase', orientation='h',
                             color='Fréquence', color_continuous_scale='Reds',
                             template="plotly_white")
            fig_neg.update_layout(
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_neg, use_container_width=True)

    with tab3:
        if country_col and country_col in df.columns:
            st.subheader("🌍 Provenance des Avis")
            country_data = df[country_col].value_counts().reset_index()
            fig_pie = px.pie(country_data, values='count', names=country_col, hole=.4, template="plotly_white")
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)