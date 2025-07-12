import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from datetime import datetime
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time

# Configuration initiale
st.set_page_config(
    page_title="Arxiv Research Assistant",
    layout="wide",
    page_icon="🔬"
)

# Charger les ressources
@st.cache_resource(ttl=3600)
def load_resources():
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
    
    # Nettoyer les auteurs
    authors = authors_df['author_name'].dropna().astype(str).tolist()
    
    # Créer un vecteur TF-IDF pour les auteurs
    if authors:
        vectorizer = TfidfVectorizer()
        author_vectors = vectorizer.fit_transform(authors)
    else:
        vectorizer = None
        author_vectors = None
    
    # Charger les catégories
    conn = sqlite3.connect(os.path.join(BASE_DIR, "arxiv_relational.db"))
    categories_df = pd.read_sql_query("SELECT DISTINCT categories FROM articles", conn)
    conn.close()
    
    all_categories = categories_df['categories'].dropna().unique()
    
    return index, article_ids, model, nlp, authors, author_vectors, vectorizer, all_categories

index, article_ids, model, nlp, authors, author_vectors, vectorizer, all_categories = load_resources()

# Créer une nouvelle connexion SQLite pour chaque thread
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "arxiv_relational.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

# Cache pour les recherches
search_cache = {}

def cached_semantic_search(query, filters):
    """Recherche avec cache basé sur la signature de la requête"""
    query_signature = hashlib.md5((query + str(filters)).encode()).hexdigest()
    if query_signature in search_cache:
        return search_cache[query_signature]
    
    # Nouvelle recherche
    result_ids, scores = semantic_search(query, k=50)
    results_df = fetch_article_details(result_ids)
    
    # Appliquer les filtres
    year_range = filters.get('year_range')
    if year_range:
        results_df = results_df[
            (results_df['year'] >= year_range[0]) & 
            (results_df['year'] <= year_range[1])
        ]
    
    domain = filters.get('domain')
    if domain:
        results_df = results_df[
            results_df['categories'].str.contains(domain, na=False)
        ]
    
    author = filters.get('author')
    if author:
        results_df = results_df[
            results_df['authors'].str.contains(author, case=False, na=False)
        ]
    
    # Ajouter les scores de pertinence
    scores_df = pd.DataFrame({
        'article_id': result_ids,
        'similarity_score': scores
    })
    
    results_df = pd.merge(
        results_df, 
        scores_df, 
        on='article_id', 
        how='inner'
    ).sort_values('similarity_score', ascending=False)
    
    search_cache[query_signature] = results_df
    return results_df

# Analyse contextuelle
def contextual_query_understanding(query):
    """Analyse contextuelle avancée avec reconnaissance d'entités"""
    doc = nlp(query)
    entities = {
        "domains": [],
        "time_periods": [],
        "authors": []
    }
    
    # Détection d'entités spécifiques
    for ent in doc.ents:
        if ent.label_ == "DATE" and re.match(r'^\d{4}$', ent.text):
            entities["time_periods"].append(int(ent.text))
        elif ent.label_ == "PERSON":
            entities["authors"].append(ent.text)
    
    # Détection d'intention
    intent = "articles"
    text = query.lower()
    
    if "author" in text or "researcher" in text or "who" in text:
        intent = "authors"
    elif "trend" in text or "evolution" in text or "over time" in text:
        intent = "trends"
    elif "collaboration" in text or "co-author" in text or "team" in text:
        intent = "collaborations"
    
    return {
        "intent": intent,
        "entities": entities
    }

# Fonctions de recherche
def semantic_search(query, k=10):
    """Recherche sémantique dans les articles"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    result_ids = [article_ids[i] for i in indices[0]]
    return result_ids, distances[0]

def find_similar_authors(author_name):
    """Trouver des auteurs similaires avec TF-IDF"""
    if vectorizer is None or author_vectors is None:
        return author_name
    
    try:
        query_vec = vectorizer.transform([author_name])
        similarities = cosine_similarity(query_vec, author_vectors)
        most_similar_idx = similarities.argmax()
        return authors[most_similar_idx]
    except:
        return author_name

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
        conditions = []
        params = []
        
        if domain:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{domain}%')
        
        if year_range:
            conditions.append("a.year BETWEEN ? AND ?")
            params.extend([year_range[0], year_range[1]])
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
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

def get_collaborators(author_name):
    """Récupérer les collaborateurs d'un auteur"""
    conn = get_db_connection()
    try:
        query = """
            SELECT a2.author_name, COUNT(*) as collab_count
            FROM article_author aa1
            JOIN article_author aa2 ON aa1.article_id = aa2.article_id
            JOIN authors a1 ON aa1.author_id = a1.author_id
            JOIN authors a2 ON aa2.author_id = a2.author_id
            WHERE a1.author_name = ? AND a2.author_name != ?
            GROUP BY a2.author_name
            ORDER BY collab_count DESC
            LIMIT 5
        """
        return pd.read_sql_query(query, conn, params=(author_name, author_name))
    finally:
        conn.close()

# Visualisations
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
        height=500
    )
    
    # Ajouter une régression polynomiale
    if len(trends_df) > 2:
        trend_line = np.polyfit(trends_df['year'], trends_df['paper_count'], 1)
        poly = np.poly1d(trend_line)
        trends_df['trend'] = poly(trends_df['year'])
        fig.add_trace(go.Scatter(
            x=trends_df['year'], 
            y=trends_df['trend'], 
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
    
    return fig

def create_collaboration_chart(author_name):
    """Crée un graphique des collaborateurs"""
    collaborators = get_collaborators(author_name)
    if collaborators.empty:
        return None
        
    fig = px.bar(
        collaborators, 
        x='collab_count', 
        y='author_name',
        orientation='h',
        title=f'Top Collaborators of {author_name}',
        labels={'author_name': 'Collaborator', 'collab_count': 'Joint Papers'}
    )
    fig.update_layout(
        yaxis_title='Collaborator',
        xaxis_title='Number of Joint Papers',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# Génération de réponses
def generate_insightful_response(analysis, results):
    """Génère des réponses structurées avec insights contextuels"""
    response = {"summary": "", "insights": [], "recommendations": []}
    
    if analysis["intent"] == "authors":
        if results.empty:
            response["summary"] = "No authors found matching your query."
            return response
            
        top_authors = results.head(3)
        domain_info = f" in {analysis['entities']['domains'][0]}" if analysis['entities'].get('domains') else ""
        
        response["summary"] = f"## Top Authors{domain_info}\n\nHere are the most prolific authors based on your query:"
        
        for i, row in top_authors.iterrows():
            collaborators = get_collaborators(row["author_name"])
            top_collabs = ", ".join(collaborators["author_name"].head(3).tolist()) if not collaborators.empty else "None"
            
            response["insights"].append({
                "name": row["author_name"],
                "paper_count": row["paper_count"],
                "top_collaborators": top_collabs
            })
            
            response["recommendations"].append(
                f"Explore collaborations with **{row['author_name']}** and their network including {top_collabs}"
            )
    
    elif analysis["intent"] == "trends":
        if results.empty:
            response["summary"] = "No trends data found for your query."
            return response
            
        domain_info = f" in {analysis['entities']['domains'][0]}" if analysis['entities'].get('domains') else ""
        response["summary"] = f"## Publication Trends{domain_info}\n\nResearch evolution over time:"
        
        # Analyse de tendance
        if len(results) > 1:
            first = results.iloc[0]['paper_count']
            last = results.iloc[-1]['paper_count']
            change = ((last - first) / first) * 100 if first > 0 else 0
            trend = "increased" if change >= 0 else "decreased"
            trend_insight = f"Publications have {trend} by {abs(change):.1f}% over this period."
        else:
            trend_insight = "Insufficient data for trend analysis."
        
        response["insights"].append({
            "trend_analysis": trend_insight
        })
        
        response["recommendations"] = [
            "Focus on emerging topics showing growth in recent years",
            "Explore interdisciplinary approaches combining top domains"
        ]
    
    elif analysis["intent"] == "collaborations":
        if results.empty:
            response["summary"] = "No author data found for collaboration analysis."
            return response
            
        author_name = results.iloc[0]["author_name"]
        collaborators = get_collaborators(author_name)
        
        if collaborators.empty:
            response["summary"] = f"No collaboration data found for {author_name}."
            return response
            
        top_collabs = ", ".join(collaborators["author_name"].head(3).tolist())
        
        response["summary"] = f"## Research Collaborations for {author_name}\n\nTop collaborators:"
        response["insights"].append({
            "main_author": author_name,
            "collaborators": collaborators.to_dict('records')
        })
        
        response["recommendations"] = [
            f"**{author_name}** frequently collaborates with {top_collabs}",
            f"Consider potential collaborations with these research teams"
        ]
    
    else:  # Articles
        if results.empty:
            response["summary"] = "No articles found matching your query."
            return response
            
        domain_info = f" in {analysis['entities']['domains'][0]}" if analysis['entities'].get('domains') else ""
        year_info = f" from {analysis['entities']['time_periods'][0]}" if analysis['entities'].get('time_periods') else ""
        
        response["summary"] = f"## Relevant Articles{domain_info}{year_info}\n\nFound {len(results)} articles matching your query:"
        
        # Analyse thématique
        if not results.empty:
            all_categories = results['categories'].str.split().explode()
            category_counts = all_categories.value_counts()
            top_categories = category_counts.head(3).index.tolist()
            
            response["insights"].append({
                "top_categories": top_categories
            })
            
            response["recommendations"] = [
                f"Explore more in **{top_categories[0]}** which is the most frequent category",
                f"Review highly cited papers in **{top_categories[1]}** for foundational knowledge"
            ]
    
    return response

# Interface utilisateur
st.title("🔬 Arxiv Research Assistant")
st.caption("An AI-powered scientific discovery engine")

# Initialiser l'état de session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("⚙️ Intelligence Settings")
    analysis_depth = st.select_slider(
        "Analysis Depth", 
        options=["Standard", "Enhanced", "Advanced"],
        value="Enhanced"
    )
    
    visualization_style = st.radio(
        "Visualization Style",
        ["Scientific", "Minimal", "Interactive"],
        index=0
    )
    
    st.divider()
    st.header("🔎 Focus Filters")
    
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

# Onglets principaux
chat_tab, results_tab, insights_tab = st.tabs(["💬 Chat", "📊 Results", "💡 Insights"])

with chat_tab:
    # Afficher l'historique de conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Entrée utilisateur
    if prompt := st.chat_input("Ask a research question..."):
        # Ajouter la question à l'historique
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing deep context..."):
            # Simuler un temps de traitement
            time.sleep(1)
            
            # Analyse contextuelle
            analysis = contextual_query_understanding(prompt)
            
            # Appliquer les filtres
            filters = {
                "year_range": selected_years,
                "domain": selected_categories[0] if selected_categories else None,
                "author": author_query
            }
            
            # Gestion des différentes intentions
            if analysis["intent"] == "authors":
                domain = analysis['entities'].get('domains', [filters['domain']])[0] if analysis['entities'].get('domains') or filters['domain'] else None
                results = get_top_authors(
                    domain=domain,
                    year_range=selected_years,
                    limit=10
                )
                response_content = generate_insightful_response(analysis, results)
                
            elif analysis["intent"] == "trends":
                domain = analysis['entities'].get('domains', [filters['domain']])[0] if analysis['entities'].get('domains') or filters['domain'] else None
                results = get_category_trends(domain=domain)
                response_content = generate_insightful_response(analysis, results)
                
            elif analysis["intent"] == "collaborations":
                author_name = None
                if analysis['entities'].get('authors'):
                    author_name = analysis['entities']['authors'][0]
                elif author_query:
                    author_name = author_query
                
                if author_name:
                    corrected_author = find_similar_authors(author_name)
                    results = pd.DataFrame({
                        "author_name": [corrected_author],
                        "paper_count": [1]  # Valeur factice pour la structure
                    })
                    response_content = generate_insightful_response(analysis, results)
                else:
                    response_content = {"summary": "Please specify an author to analyze collaborations."}
                    results = pd.DataFrame()
                    
            else:  # Recherche d'articles
                results = cached_semantic_search(prompt, filters)
                response_content = generate_insightful_response(analysis, results)
            
            # Stocker les résultats pour les autres onglets
            st.session_state.results = results
            st.session_state.analysis = analysis
            st.session_state.response_content = response_content
            
            # Formater la réponse pour le chat
            response_text = response_content["summary"] + "\n\n"
            
            if response_content.get("insights"):
                for insight in response_content["insights"]:
                    if "top_collaborators" in insight:
                        response_text += f"- **{insight['name']}**: {insight['paper_count']} papers, Top collaborators: {insight['top_collaborators']}\n"
                    elif "trend_analysis" in insight:
                        response_text += f"- Trend Analysis: {insight['trend_analysis']}\n"
            
            if response_content.get("recommendations"):
                response_text += "\n**Recommendations**:\n"
                for rec in response_content["recommendations"]:
                    response_text += f"- {rec}\n"
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text
            })
            
            # Re-exécuter pour afficher la nouvelle réponse
            st.rerun()

with results_tab:
    if "results" in st.session_state and not st.session_state.results.empty:
        results = st.session_state.results
        analysis = st.session_state.analysis
        
        if analysis["intent"] == "authors":
            st.subheader("Top Authors Analysis")
            fig = plot_author_distribution(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les collaborations pour le premier auteur
            if not results.empty:
                author_name = results.iloc[0]["author_name"]
                fig_collab = create_collaboration_chart(author_name)
                if fig_collab:
                    st.subheader(f"Collaboration Network of {author_name}")
                    st.plotly_chart(fig_collab, use_container_width=True)
        
        elif analysis["intent"] == "trends":
            st.subheader("Publication Trends")
            fig = plot_trends(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis["intent"] == "collaborations" and "main_author" in st.session_state.response_content.get("insights", [{}])[0]:
            author_name = st.session_state.response_content["insights"][0]["main_author"]
            fig_collab = create_collaboration_chart(author_name)
            if fig_collab:
                st.subheader(f"Collaboration Analysis for {author_name}")
                st.plotly_chart(fig_collab, use_container_width=True)
        
        else:  # Articles
            st.subheader(f"🔍 Top {len(results)} Relevant Articles")
            
            for i, row in results.iterrows():
                with st.expander(f"**{row['title']}** ({row['year']})", expanded=(i < 2)):
                    st.markdown(f"**Authors:** {row['authors']}")
                    st.markdown(f"**Categories:** {row['categories']}")
                    
                    # Afficher l'abstract avec une limite de caractères
                    abstract = row['abstract']
                    if len(abstract) > 500:
                        st.markdown(f"**Abstract:** {abstract[:500]}...")
                    else:
                        st.markdown(f"**Abstract:** {abstract}")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Relevance Score", f"{row['similarity_score']:.2f}")
                    with col2:
                        if row.get('doi'):
                            st.markdown(f"[📄 View Full Paper](https://doi.org/{row['doi']})")
                    
                    st.divider()
    else:
        st.info("Run a search to see results here")

with insights_tab:
    if "response_content" in st.session_state:
        response = st.session_state.response_content
        
        st.subheader("Key Insights")
        st.markdown(response["summary"])
        
        if response.get("insights"):
            for insight in response["insights"]:
                with st.expander("🔍 Detailed Analysis", expanded=True):
                    if "top_categories" in insight:
                        st.write("**Top Research Categories:**")
                        st.write(", ".join(insight["top_categories"]))
                    
                    elif "top_collaborators" in insight:
                        st.write(f"**{insight['name']}** has published **{insight['paper_count']} papers**")
                        st.write(f"**Top collaborators:** {insight['top_collaborators']}")
                    
                    elif "trend_analysis" in insight:
                        st.write(insight["trend_analysis"])
                    
                    elif "collaborators" in insight:
                        st.write(f"**Top Collaborators of {insight['main_author']}:**")
                        for collab in insight["collaborators"]:
                            st.write(f"- {collab['author_name']}: {collab['collab_count']} joint papers")
        
        if response.get("recommendations"):
            st.subheader("Actionable Recommendations")
            for rec in response["recommendations"]:
                st.info(rec)
    else:
        st.info("Run a search to see insights here")

# Exemples de requêtes
with st.expander("💡 Example Queries", expanded=False):
    st.markdown("""
    **Try these intelligent queries:**
    
    - "Top authors in machine learning"
    - "Recent breakthroughs in quantum computing"
    - "Trending research topics in NLP"
    - "Papers by Yann LeCun about convolutional networks"
    - "Publication trends in AI between 2020-2023"
    - "Collaborators of Geoffrey Hinton"
    - "Seminal papers about transformers"
    """)

# Pied de page
st.divider()
st.caption("© 2025 Arxiv Research Assistant | Powered by Semantic Search and NLP")