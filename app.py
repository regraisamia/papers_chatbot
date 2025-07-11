import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from datetime import datetime
import os
import spacy
from collections import Counter

# Configuration initiale
st.set_page_config(page_title="Arxiv Research Assistant", layout="wide")

# Charger les ressources
@st.cache_resource
def load_resources():
    # Chemin absolu pour éviter les problèmes
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Charger l'index FAISS
    faiss_path = os.path.join(BASE_DIR, "faiss_index.bin")
    index = faiss.read_index(faiss_path)
    
    # Charger les IDs d'articles
    ids_path = os.path.join(BASE_DIR, "article_ids.pkl")
    with open(ids_path, "rb") as f:
        article_ids = pickle.load(f)
    
    # Charger le modèle de langage
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Charger le modèle NLP
    try:
        nlp = spacy.load("en_core_web_md")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load("en_core_web_md")
    
    return index, article_ids, model, nlp

index, article_ids, model, nlp = load_resources()

# Créer une nouvelle connexion SQLite pour chaque thread
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "arxiv_relational.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

# Fonctions utilitaires
def preprocess_query(query):
    """Nettoyer et normaliser la requête"""
    return query.strip()

def analyze_query(query):
    """Analyse la requête pour détecter l'intention et les entités"""
    # Liste d'intentions avec des motifs clés
    author_intent_patterns = [
        r"top (authors|researchers|scientists)",
        r"most (prolific|published) authors",
        r"who (published|has) the most papers",
        r"leading authors in",
        r"authors with (many|a lot of) papers",
        r"rank authors by publications"
    ]
    
    trend_intent_patterns = [
        r"trends? (in|over time)",
        r"evolution of publications",
        r"growth of research in",
        r"publication (history|timeline)",
        r"how many papers (published|over time)"
    ]
    
    # Vérification par motif regex
    query_lower = query.lower()
    intent = "articles"  # Par défaut
    
    for pattern in author_intent_patterns:
        if re.search(pattern, query_lower):
            intent = "authors"
            break
            
    for pattern in trend_intent_patterns:
        if re.search(pattern, query_lower):
            intent = "trends"
            break
    
    # Détection des entités
    entities = {
        "year": None,
        "year_range": None,
        "field": None,
        "limit": 10
    }
    
    # Détection des années
    year_matches = re.findall(r'\b(20[0-2][0-9])\b', query)
    if year_matches:
        entities['year'] = int(year_matches[0])
    
    # Détection des plages d'années
    year_range_match = re.findall(r'(\d{4})\s*[-to]+\s*(\d{4})', query)
    if year_range_match:
        start, end = year_range_match[0]
        entities['year_range'] = (int(start), int(end))
    
    # Détection des limites numériques
    num_match = re.search(r'(top|first|leading)\s+(\d+)', query_lower)
    if num_match:
        entities['limit'] = int(num_match.group(2))
    
    # Mapping intelligent des domaines
    field_mapping = {
        "machine learning": "cs.LG",
        "ml": "cs.LG",
        "deep learning": "cs.LG",
        "ai": "cs.AI",
        "artificial intelligence": "cs.AI",
        "nlp": "cs.CL",
        "natural language processing": "cs.CL",
        "computer vision": "cs.CV",
        "cv": "cs.CV",
        "robotics": "cs.RO",
        "data science": "cs.LG",
        "neural networks": "cs.LG"
    }
    
    for term, code in field_mapping.items():
        if term in query_lower:
            entities['field'] = code
            break
    
    return intent, entities

def semantic_search(query, k=10):
    """Recherche sémantique dans les articles"""
    # Encoder la requête
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Recherche dans FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Récupérer les IDs d'articles
    result_ids = [article_ids[i] for i in indices[0]]
    
    return result_ids, distances[0]

def fetch_article_details(article_ids):
    """Récupérer les détails des articles depuis la base de données"""
    if not article_ids:
        return pd.DataFrame()
    
    conn = get_db_connection()
    try:
        placeholders = ', '.join(['?'] * len(article_ids))
        query = f"""
            SELECT article_id, title, authors, abstract, year, categories, doi, license
            FROM articles
            WHERE article_id IN ({placeholders})
        """
        return pd.read_sql_query(query, conn, params=article_ids)
    finally:
        conn.close()

def get_top_authors(field=None, year_range=None, limit=10):
    """Récupérer les auteurs les plus prolifiques"""
    conn = get_db_connection()
    try:
        # Construire la requête dynamiquement
        conditions = []
        params = []
        
        if field:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{field}%')
        
        if year_range:
            conditions.append("a.year BETWEEN ? AND ?")
            params.extend([year_range[0], year_range[1]])
        
        # Construire la clause WHERE
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Requête corrigée avec jointure appropriée
        query = f"""
            SELECT au.author_name, COUNT(DISTINCT aa.article_id) as paper_count
            FROM article_author aa
            JOIN authors au ON aa.author_id = au.author_id
            JOIN articles a ON aa.article_id = a.article_id
            {where_clause}
            GROUP BY au.author_name
            ORDER BY paper_count DESC
            LIMIT ?
        """
        params.append(limit)
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_category_trends(category=None):
    """Récupérer les tendances par année pour une catégorie"""
    conn = get_db_connection()
    try:
        conditions = []
        params = []
        
        if category:
            conditions.append("categories LIKE ?")
            params.append(f'%{category}%')
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT year, COUNT(*) as paper_count
            FROM articles
            {where_clause}
            GROUP BY year
            ORDER BY year
        """
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def generate_author_response(authors_df, field=None, year_range=None):
    """Générer une réponse pour les requêtes sur les auteurs"""
    if authors_df.empty:
        return "I couldn't find any authors matching your query. Please try different filters."
    
    # Créer une description contextuelle
    context_parts = []
    if field:
        context_parts.append(f"in {field.replace('cs.', '').upper()}")
    if year_range:
        context_parts.append(f"between {year_range[0]} and {year_range[1]}")
    
    context_str = " " + " ".join(context_parts) if context_parts else ""
    
    response = f"## Top Authors{context_str}\n\n"
    response += "Here are the most prolific authors based on your query:\n\n"
    
    for i, row in authors_df.iterrows():
        response += f"{i+1}. **{row['author_name']}** - {row['paper_count']} papers\n"
    
    return response

def generate_trend_response(trends_df, category=None):
    """Générer une réponse pour les tendances"""
    if trends_df.empty:
        return "I couldn't find any trends data matching your query."
    
    context = f" in {category.replace('cs.', '').upper()}" if category else ""
    
    response = f"## Publication Trends{context}\n\n"
    response += "Number of publications per year:\n\n"
    
    for _, row in trends_df.iterrows():
        response += f"- **{row['year']}**: {row['paper_count']} papers\n"
    
    return response

def plot_author_distribution(authors_df):
    """Visualisation de la distribution des auteurs"""
    if authors_df.empty:
        return None
        
    # Limiter à 20 auteurs max pour la lisibilité
    if len(authors_df) > 20:
        authors_df = authors_df.head(20)
        
    fig = px.bar(
        authors_df, 
        x='author_name', 
        y='paper_count',
        title='Top Authors by Publication Count',
        labels={'author_name': 'Author', 'paper_count': 'Number of Papers'},
        color='paper_count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title='Author',
        yaxis_title='Number of Papers',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    return fig

def plot_trends(trends_df):
    """Visualisation des tendances temporelles"""
    if trends_df.empty:
        return None
        
    fig = px.line(
        trends_df, 
        x='year', 
        y='paper_count',
        title='Publication Trends Over Time',
        labels={'year': 'Year', 'paper_count': 'Number of Publications'},
        markers=True,
        line_shape='linear'
    )
    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Publications',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Ajouter une régression polynomiale pour montrer la tendance
    if len(trends_df) > 2:
        trend_line = np.polyfit(trends_df['year'], trends_df['paper_count'], 2)
        poly = np.poly1d(trend_line)
        trends_df['trend'] = poly(trends_df['year'])
        fig.add_scatter(
            x=trends_df['year'], 
            y=trends_df['trend'], 
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        )
    
    return fig

# Interface utilisateur
st.title("🧠 Arxiv Research Assistant")
st.caption("An intelligent chatbot for scientific research")

# Barre latérale pour les filtres
with st.sidebar:
    st.header("🔍 Filters")
    
    # Filtre par année
    min_year, max_year = 2020, datetime.now().year
    selected_years = st.slider(
        "Publication Year", 
        min_value=min_year, 
        max_value=max_year, 
        value=(min_year, max_year)
    )
    
    # Filtre par catégorie
    conn = get_db_connection()
    try:
        categories = pd.read_sql_query(
            "SELECT DISTINCT categories FROM articles", 
            conn
        )['categories'].unique()
    finally:
        conn.close()
    
    main_categories = set()
    for cat_list in categories:
        if cat_list:
            main_categories.update(cat_list.split())
    
    # Filtrer les catégories CS principales
    cs_categories = sorted([cat for cat in main_categories if cat.startswith('cs.')])
    
    selected_categories = st.multiselect(
        "Categories", 
        cs_categories,
        default=[]
    )
    
    # Filtre par auteur
    author_query = st.text_input("Search by Author")

# Zone de recherche principale
query = st.text_input(
    "Ask a research question:", 
    placeholder="e.g., 'Top authors in machine learning between 2020-2022'",
    key="search_input"
)

search_button = st.button("Search")

if search_button or st.session_state.get('auto_search', False):
    if not query:
        st.warning("Please enter a search query")
        st.stop()
    
    # Analyser la requête
    intent, entities = analyze_query(query)
    
    # Debug: Afficher l'intention détectée
    st.info(f"Detected intent: **{intent}**")
    
    # Appliquer les filtres supplémentaires de la barre latérale
    year_range = None
    if selected_years != (min_year, max_year):
        year_range = (selected_years[0], selected_years[1])
    
    # Gestion des différentes intentions
    if intent == "authors":
        # Déterminer le champ de recherche
        field = entities.get('field')
        if not field and selected_categories:
            field = selected_categories[0]
        
        # Déterminer la plage d'années
        if entities.get('year_range'):
            year_range = entities['year_range']
        elif entities.get('year'):
            year_range = (entities['year'], entities['year'])
        
        # Obtenir le nombre de résultats
        limit = entities.get('limit', 10)
        
        # Récupérer les auteurs
        authors_df = get_top_authors(
            field=field, 
            year_range=year_range,
            limit=limit
        )
        
        # Générer et afficher la réponse
        response = generate_author_response(authors_df, field, year_range)
        st.markdown(response)
        
        # Afficher le graphique
        if not authors_df.empty:
            fig = plot_author_distribution(authors_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No authors found matching your criteria")
    
    elif intent == "trends":
        # Déterminer le champ de recherche
        field = entities.get('field')
        if not field and selected_categories:
            field = selected_categories[0]
        
        # Récupérer les tendances
        trends_df = get_category_trends(category=field)
        
        # Générer et afficher la réponse
        response = generate_trend_response(trends_df, field)
        st.markdown(response)
        
        # Afficher le graphique
        if not trends_df.empty:
            fig = plot_trends(trends_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trend data found for your query")
    
    else:  # Recherche d'articles par défaut
        # Recherche sémantique
        result_ids, scores = semantic_search(query, k=50)
        
        if result_ids:
            results_df = fetch_article_details(result_ids)
            
            # Appliquer les filtres supplémentaires
            if year_range:
                results_df = results_df[
                    (results_df['year'] >= year_range[0]) & 
                    (results_df['year'] <= year_range[1])
                ]
            
            if selected_categories:
                category_condition = "|".join(selected_categories)
                results_df = results_df[
                    results_df['categories'].str.contains(category_condition)
                ]
            
            if author_query:
                results_df = results_df[
                    results_df['authors'].str.contains(author_query, case=False)
                ]
            
            # Ajouter les scores de pertinence
            if not results_df.empty:
                # Conserver l'ordre original des résultats
                results_df = results_df.set_index('article_id')
                results_df = results_df.loc[result_ids]
                results_df = results_df.reset_index()
                
                # Ajouter les scores
                results_df['similarity_score'] = scores[:len(results_df)]
                results_df = results_df.sort_values('similarity_score', ascending=False)
                
                # Afficher les résultats
                st.subheader(f"Found {len(results_df)} relevant articles:")
                
                for i, row in results_df.iterrows():
                    with st.expander(f"**{row['title']}** ({row['year']})", expanded=False):
                        st.markdown(f"**Authors:** {row['authors']}")
                        st.markdown(f"**Categories:** {row['categories']}")
                        st.markdown(f"**Abstract:** {row['abstract'][:500]}...")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Similarity", f"{row['similarity_score']:.2f}")
                        with col2:
                            if row['doi']:
                                st.markdown(f"[View Full Paper](https://doi.org/{row['doi']})")
                        
                        st.divider()
            else:
                st.warning("No articles found matching your query")
        else:
            st.warning("No articles found matching your query")

# Section de documentation
with st.expander("📖 How to use this assistant"):
    st.markdown("""
    **The Arxiv Research Assistant understands natural language queries and can help you:**
    
    - Find relevant scientific papers
    - Discover top authors in specific fields
    - Analyze publication trends over time
    
    **Examples of queries you can try:**
    
    - "Top authors in machine learning"
    - "Most published researchers in NLP between 2020-2022"
    - "Publication trends in computer vision"
    - "Recent papers about transformer models"
    - "Important papers about GANs in computer vision"
    
    **Tips for better results:**
    
    1. Be specific: "machine learning" vs "deep learning for image recognition"
    2. Use time filters: "papers from 2021", "trends since 2020"
    3. Specify fields: "in computer vision", "about natural language processing"
    4. Use the sidebar filters to refine results
    """)

# Pied de page
st.divider()
st.caption("© 2023 Arxiv Research Assistant | Powered by Semantic Search and NLP")

# Initialiser la recherche automatique après la première exécution
if search_button:
    st.session_state.auto_search = True