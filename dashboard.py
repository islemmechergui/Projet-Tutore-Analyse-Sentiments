import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def run_dashboard(df, review_text_col, country_col):
    """
    Dashboard interactif pour l'analyse des sentiments Amazon
    """
    st.title("📊 Dashboard Interactif")
    
    st.markdown("""
    Ce dashboard offre une vue d'ensemble interactive des données d'analyse de sentiments.
    """)
    
    # Vérifications
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible")
        return
    
    # =========================
    # Métriques principales
    # =========================
    st.subheader("📈 Métriques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df)
    positifs = (df["label"] == 1).sum() if "label" in df.columns else 0
    neutres = (df["label"] == 0).sum() if "label" in df.columns else 0
    negatifs = (df["label"] == -1).sum() if "label" in df.columns else 0
    
    col1.metric("📊 Total Reviews", f"{total_reviews:,}")
    col2.metric("😊 Positifs", f"{positifs:,}", f"{100*positifs/total_reviews:.1f}%" if total_reviews > 0 else "0%")
    col3.metric("😐 Neutres", f"{neutres:,}", f"{100*neutres/total_reviews:.1f}%" if total_reviews > 0 else "0%")
    col4.metric("😞 Négatifs", f"{negatifs:,}", f"{100*negatifs/total_reviews:.1f}%" if total_reviews > 0 else "0%")
    
    # =========================
    # Graphiques interactifs
    # =========================
    st.markdown("---")
    
    # Répartition des sentiments (Donut Chart)
    if "label" in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Répartition des Sentiments")
            sentiment_counts = df["label"].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Positif", "Neutre", "Négatif"],
                values=[sentiment_counts.get(1, 0), sentiment_counts.get(0, 0), sentiment_counts.get(-1, 0)],
                hole=.4,
                marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"])
            )])
            fig_pie.update_layout(
                showlegend=True,
                height=400,
                title_text="Distribution des sentiments"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Distribution des ratings
        with col2:
            if "numeric_rating" in df.columns:
                st.subheader("⭐ Distribution des Notes")
                rating_counts = df["numeric_rating"].value_counts().sort_index()
                
                fig_bar = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    labels={"x": "Note", "y": "Nombre d'avis"},
                    color=rating_counts.values,
                    color_continuous_scale="Blues"
                )
                fig_bar.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_title="Note",
                    yaxis_title="Nombre d'avis"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # =========================
    # Analyse par pays
    # =========================
    st.markdown("---")
    st.subheader("🌍 Analyse Géographique")
    
    if country_col and country_col in df.columns:
        # Top 10 pays
        top_countries = df[country_col].value_counts().head(10)
        
        fig_countries = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation='h',
            labels={"x": "Nombre d'avis", "y": "Pays"},
            title="Top 10 Pays",
            color=top_countries.values,
            color_continuous_scale="Viridis"
        )
        fig_countries.update_layout(
            showlegend=False,
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_countries, use_container_width=True)
        
        # Sentiments par pays (Top 5)
        if "label" in df.columns:
            st.subheader("📊 Sentiments par Pays (Top 5)")
            top_5_countries = df[country_col].value_counts().head(5).index
            df_top5 = df[df[country_col].isin(top_5_countries)]
            
            sentiment_by_country = df_top5.groupby([country_col, "label"]).size().reset_index(name="count")
            sentiment_by_country["sentiment"] = sentiment_by_country["label"].map({
                -1: "Négatif",
                0: "Neutre",
                1: "Positif"
            })
            
            fig_stacked = px.bar(
                sentiment_by_country,
                x=country_col,
                y="count",
                color="sentiment",
                title="Distribution des sentiments par pays",
                color_discrete_map={
                    "Positif": "#10b981",
                    "Neutre": "#f59e0b",
                    "Négatif": "#ef4444"
                },
                barmode="stack"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
    else:
        st.info("ℹ️ Aucune information géographique disponible")
    
    # =========================
    # Evolution temporelle
    # =========================
    st.markdown("---")
    st.subheader("📅 Évolution Temporelle")
    
    date_col = "Review Date" if "Review Date" in df.columns else None
    
    if date_col:
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
        df_temp = df_temp.dropna(subset=[date_col])
        
        if len(df_temp) > 0:
            df_temp["Année-Mois"] = df_temp[date_col].dt.to_period("M").astype(str)
            
            # Evolution du nombre d'avis
            time_series = df_temp.groupby("Année-Mois").size().reset_index(name="Nombre d'avis")
            
            fig_time = px.line(
                time_series,
                x="Année-Mois",
                y="Nombre d'avis",
                title="Évolution du nombre d'avis dans le temps",
                markers=True
            )
            fig_time.update_layout(
                xaxis_title="Période",
                yaxis_title="Nombre d'avis",
                hovermode="x unified"
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Evolution des sentiments si disponible
            if "label" in df_temp.columns:
                sentiment_time = df_temp.groupby(["Année-Mois", "label"]).size().reset_index(name="count")
                sentiment_time["sentiment"] = sentiment_time["label"].map({
                    -1: "Négatif",
                    0: "Neutre",
                    1: "Positif"
                })
                
                fig_sentiment_time = px.line(
                    sentiment_time,
                    x="Année-Mois",
                    y="count",
                    color="sentiment",
                    title="Évolution des sentiments dans le temps",
                    markers=True,
                    color_discrete_map={
                        "Positif": "#10b981",
                        "Neutre": "#f59e0b",
                        "Négatif": "#ef4444"
                    }
                )
                fig_sentiment_time.update_layout(
                    xaxis_title="Période",
                    yaxis_title="Nombre d'avis",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_sentiment_time, use_container_width=True)
        else:
            st.warning("⚠️ Aucune date valide trouvée dans le dataset")
    else:
        st.info("ℹ️ Aucune information temporelle disponible")
    
    # =========================
    # Statistiques avancées
    # =========================
    st.markdown("---")
    st.subheader("📊 Statistiques Détaillées")
    
    with st.expander("📈 Voir les statistiques détaillées"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Informations du Dataset**")
            st.write(f"• Nombre total de lignes : {len(df):,}")
            st.write(f"• Nombre de colonnes : {len(df.columns)}")
            if review_text_col:
                avg_length = df[review_text_col].astype(str).str.len().mean()
                st.write(f"• Longueur moyenne des avis : {avg_length:.0f} caractères")
        
        with col2:
            if "label" in df.columns:
                st.markdown("**🎯 Distribution des Sentiments**")
                st.write(f"• Positifs : {positifs:,} ({100*positifs/total_reviews:.1f}%)")
                st.write(f"• Neutres : {neutres:,} ({100*neutres/total_reviews:.1f}%)")
                st.write(f"• Négatifs : {negatifs:,} ({100*negatifs/total_reviews:.1f}%)")
    
    st.success("✅ Dashboard chargé avec succès !")
