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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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
    
    # Charger la liste des auteurs
    conn = sqlite3.connect(os.path.join(BASE_DIR, "arxiv_relational.db"))
    authors_df = pd.read_sql_query("SELECT DISTINCT author_name FROM authors", conn)
    conn.close()
    
    # Nettoyer les auteurs - supprimer les valeurs nulles
    authors = authors_df['author_name'].dropna().astype(str).tolist()
    
    # Créer un vecteur TF-IDF pour les auteurs
    if authors:  # Vérifier que la liste n'est pas vide
        vectorizer = TfidfVectorizer()
        author_vectors = vectorizer.fit_transform(authors)
    else:
        vectorizer = None
        author_vectors = None
    
    # Charger les catégories
    conn = sqlite3.connect(os.path.join(BASE_DIR, "arxiv_relational.db"))
    categories = pd.read_sql_query("SELECT DISTINCT categories FROM articles", conn)['categories'].dropna().unique()
    conn.close()
    
    return index, article_ids, model, nlp, authors, author_vectors, vectorizer, categories

index, article_ids, model, nlp, authors, author_vectors, vectorizer, all_categories = load_resources()

# Créer une nouvelle connexion SQLite pour chaque thread
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "arxiv_relational.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

# Fonctions d'intelligence avancée
def advanced_query_analysis(query):
    """Analyse avancée de la requête avec NLP"""
    doc = nlp(query.lower())
    
    # Détection d'intention
    intent = "articles"
    intent_keywords = {
        "authors": ["author", "researcher", "scientist", "who", "prolific", "published", "most", "top", "leading"],
        "trends": ["trend", "evolution", "growth", "over time", "history", "development", "timeline"],
        "categories": ["category", "field", "domain", "area", "discipline"],
        "collaborations": ["collaboration", "co-author", "network", "team", "partnership"]
    }
    
    for token in doc:
        for intent_type, keywords in intent_keywords.items():
            if token.lemma_ in keywords:
                intent = intent_type
                break
    
    # Extraction d'entités
    entities = {
        "years": [],
        "year_range": None,
        "authors": [],
        "categories": [],
        "limit": 10
    }
    
    # Extraction des années
    for ent in doc.ents:
        if ent.label_ == "DATE" and re.match(r'^\d{4}$', ent.text):
            entities["years"].append(int(ent.text))
        elif ent.label_ == "PERSON":
            entities["authors"].append(ent.text)
    
    # Extraction des plages d'années
    year_range_match = re.search(r'(\d{4})\s*[-to]+\s*(\d{4})', query)
    if year_range_match:
        entities["year_range"] = (int(year_range_match.group(1)), int(year_range_match.group(2)))
    
    # Extraction des limites numériques
    num_match = re.search(r'(top|first|leading)\s+(\d+)', query.lower())
    if num_match:
        entities["limit"] = int(num_match.group(2))
    
    # Mapping des catégories
    category_mapping = {
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
    
    for token in doc:
        for term, code in category_mapping.items():
            if token.lemma_ in term.split():
                entities["categories"].append(code)
    
    return intent, entities

def find_similar_authors(author_name):
    """Trouver des auteurs similaires avec TF-IDF"""
    if vectorizer is None or author_vectors is None:
        return author_name  # Retourner l'original si pas de vecteurs
    
    try:
        query_vec = vectorizer.transform([author_name])
        similarities = cosine_similarity(query_vec, author_vectors)
        most_similar_idx = similarities.argmax()
        return authors[most_similar_idx]
    except:
        return author_name  # Retourner l'original en cas d'erreur

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

def get_top_authors(domain=None, year_range=None, limit=10):
    """Récupérer les auteurs les plus prolifiques"""
    conn = get_db_connection()
    try:
        # Construire la requête dynamiquement
        conditions = []
        params = []
        
        if domain:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{domain}%')
        
        if year_range:
            conditions.append("a.year BETWEEN ? AND ?")
            params.extend([year_range[0], year_range[1]])
        
        # Construire la clause WHERE
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Requête
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

def get_category_trends(domain=None):
    """Récupérer les tendances par année pour un domaine"""
    conn = get_db_connection()
    try:
        conditions = []
        params = []
        
        if domain:
            conditions.append("categories LIKE ?")
            params.append(f'%{domain}%')
        
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

def generate_insightful_response(intent, results, entities):
    """Générer une réponse intelligente sans GPT"""
    if intent == "authors":
        if results.empty:
            return "I couldn't find any authors matching your query. Please try different filters."
        
        domain_info = f" in {entities['categories'][0]}" if entities['categories'] else ""
        year_info = f" between {entities['year_range'][0]} and {entities['year_range'][1]}" if entities['year_range'] else ""
        
        response = f"## Top Authors{domain_info}{year_info}\n\n"
        response += "Here are the most prolific authors based on your query:\n\n"
        
        for i, row in results.iterrows():
            response += f"{i+1}. **{row['author_name']}** - {row['paper_count']} papers\n"
        
        return response
    
    elif intent == "trends":
        if results.empty:
            return "I couldn't find any trends data matching your query."
        
        domain_info = f" in {entities['categories'][0]}" if entities['categories'] else ""
        response = f"## Publication Trends{domain_info}\n\n"
        response += "Number of publications per year:\n\n"
        
        for _, row in results.iterrows():
            response += f"- **{row['year']}**: {row['paper_count']} papers\n"
        
        # Analyse de tendance simple
        if len(results) > 1:
            first = results.iloc[0]['paper_count']
            last = results.iloc[-1]['paper_count']
            change = ((last - first) / first) * 100 if first > 0 else 0
            trend = "increased" if change >= 0 else "decreased"
            response += f"\nPublications have {trend} by {abs(change):.1f}% over this period."
        
        return response
    
    else:  # Articles
        if results.empty:
            return "I couldn't find any articles matching your query."
        
        domain_info = f" in {entities['categories'][0]}" if entities['categories'] else ""
        year_info = f" from {entities['years'][0]}" if entities['years'] else ""
        response = f"## Relevant Articles{domain_info}{year_info}\n\n"
        response += f"Found {len(results)} articles matching your query:\n\n"
        
        # Grouper par similarité
        high_sim = results[results['similarity_score'] > 0.7]
        med_sim = results[(results['similarity_score'] > 0.5) & (results['similarity_score'] <= 0.7)]
        low_sim = results[results['similarity_score'] <= 0.5]
        
        if not high_sim.empty:
            response += "### Highly Relevant\n"
            for i, row in high_sim.iterrows():
                response += f"- **{row['title']}** ({row['year']}) - Score: {row['similarity_score']:.2f}\n"
        
        if not med_sim.empty:
            response += "\n### Moderately Relevant\n"
            for i, row in med_sim.iterrows():
                response += f"- **{row['title']}** ({row['year']}) - Score: {row['similarity_score']:.2f}\n"
        
        if not low_sim.empty:
            response += "\n### Possibly Relevant\n"
            for i, row in low_sim.iterrows():
                response += f"- **{row['title']}** ({row['year']}) - Score: {row['similarity_score']:.2f}\n"
        
        # Analyse thématique
        if not results.empty:
            categories = results['categories'].str.split().explode().value_counts()
            top_categories = categories.head(3).index.tolist()
            response += f"\nTop categories: {', '.join(top_categories)}"
        
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
        y='author_name', 
        x='paper_count',
        orientation='h',
        title='Top Authors by Publication Count',
        labels={'author_name': 'Author', 'paper_count': 'Number of Papers'},
        color='paper_count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis_title='Author',
        xaxis_title='Number of Papers',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    fig.update_yaxes(autorange="reversed")
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
st.caption("An intelligent scientific research chatbot")

# Barre latérale pour les filtres
with st.sidebar:
    st.header("🔍 Advanced Filters")
    
    # Filtre par année
    min_year, max_year = 2020, datetime.now().year
    selected_years = st.slider(
        "Publication Year", 
        min_value=min_year, 
        max_value=max_year, 
        value=(min_year, max_year)
    )
    
    # Filtre par catégorie
    main_categories = set()
    for cat_list in all_categories:
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
    "Ask any research question:", 
    placeholder="e.g., 'Who are the most influential authors in AI?', 'Show me recent breakthroughs in quantum computing'",
    key="search_input"
)

search_button = st.button("Search", type="primary")

if search_button:
    if not query:
        st.warning("Please enter a search query")
        st.stop()
    
    with st.spinner("Analyzing your question..."):
        # Analyse intelligente de la requête
        intent, entities = advanced_query_analysis(query)
        
        # Détails de l'analyse
        with st.expander("Analysis Details"):
            st.write(f"**Detected intent:** {intent}")
            st.write(f"**Extracted entities:**")
            st.json(entities)
        
        # Appliquer les filtres de la barre latérale
        year_range = (selected_years[0], selected_years[1])
        domain = entities['categories'][0] if entities['categories'] else (selected_categories[0] if selected_categories else None)
        
        # Correction du nom d'auteur si nécessaire
        if entities['authors'] and vectorizer is not None:
            corrected_author = find_similar_authors(entities['authors'][0])
            if corrected_author != entities['authors'][0]:
                st.info(f"Did you mean: {corrected_author}? Using this for search.")
                entities['authors'][0] = corrected_author
        
        # Gestion des différentes intentions
        if intent == "authors":
            # Récupérer les auteurs
            authors_df = get_top_authors(
                domain=domain,
                year_range=entities.get('year_range', year_range),
                limit=entities.get('limit', 10)
            )
            
            # Générer et afficher la réponse
            response = generate_insightful_response(intent, authors_df, entities)
            st.markdown(response)
            
            # Afficher le graphique
            if not authors_df.empty:
                fig = plot_author_distribution(authors_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No authors found matching your criteria")
        
        elif intent == "trends":
            # Récupérer les tendances
            trends_df = get_category_trends(domain=domain)
            
            # Générer et afficher la réponse
            response = generate_insightful_response(intent, trends_df, entities)
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
                if entities.get('year_range') or entities.get('years'):
                    year_range = entities.get('year_range', year_range)
                    if entities.get('years'):
                        year_range = (min(entities['years']), max(entities['years']))
                    
                    results_df = results_df[
                        (results_df['year'] >= year_range[0]) & 
                        (results_df['year'] <= year_range[1])
                    ]
                
                if domain:
                    results_df = results_df[
                        results_df['categories'].str.contains(domain)
                    ]
                
                if entities['authors']:
                    results_df = results_df[
                        results_df['authors'].str.contains(entities['authors'][0], case=False)
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
                    
                    # Générer et afficher la réponse
                    response = generate_insightful_response(intent, results_df, entities)
                    st.markdown(response)
                    
                    # Afficher les résultats détaillés
                    st.subheader(f"🔍 Top {len(results_df)} Relevant Articles")
                    
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
                                    st.markdown(f"[📄 View Full Paper](https://doi.org/{row['doi']})")
                            
                            st.divider()
                else:
                    st.warning("No articles found matching your query")
            else:
                st.warning("No articles found matching your query")

# Section d'exemples
with st.expander("💡 Example Queries"):
    st.markdown("""
    **Try these intelligent queries:**
    
    - "Who are the most influential authors in machine learning?"
    - "Show me recent breakthroughs in quantum computing"
    - "What are the trending research topics in NLP?"
    - "Find papers by Yann LeCun about convolutional networks"
    - "Compare the publication trends in AI and biology"
    - "Who collaborated the most with Geoffrey Hinton?"
    - "What are the seminal papers about transformers?"
    """)

# Pied de page
st.divider()
st.caption("© 2023 Arxiv Research Assistant | Powered by Semantic Search and NLP")