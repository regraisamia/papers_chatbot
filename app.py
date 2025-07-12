import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

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
import hashlib
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import textwrap
import uuid  # For generating unique keys

# Initial configuration
st.set_page_config(
    page_title=" Papers Research Assistant",
    layout="wide",
    page_icon="🧠"
)

# Download necessary resources
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()

# Load resources
@st.cache_resource(ttl=3600)
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load FAISS index
    faiss_path = os.path.join(BASE_DIR, "faiss_index.bin")
    index = faiss.read_index(faiss_path)
    
    # Load article IDs
    ids_path = os.path.join(BASE_DIR, "article_ids.pkl")
    with open(ids_path, "rb") as f:
        article_ids = pickle.load(f)
    
    # Load language model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load authors list
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, "arxiv_relational.db"))
        authors_df = pd.read_sql_query("SELECT DISTINCT author_name FROM authors", conn)
        conn.close()
        authors = authors_df['author_name'].dropna().astype(str).tolist()
    except:
        authors = []
    
    return index, article_ids, model, authors

index, article_ids, model, authors = load_resources()

# Create new SQLite connection
def get_db_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "arxiv_relational.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

# Cache for searches
search_cache = {}

def cached_semantic_search(query, filters, k=100):
    """Search with query signature-based caching"""
    query_signature = hashlib.md5((query + str(filters)).encode()).hexdigest()
    if query_signature in search_cache:
        return search_cache[query_signature]
    
    # New search
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    result_ids = [article_ids[i] for i in indices[0]]
    scores = distances[0]
    
    # Retrieve article details
    conn = get_db_connection()
    placeholders = ', '.join(['?'] * len(result_ids))
    query_sql = f"""
        SELECT article_id, title, authors, abstract, year, categories, doi
        FROM articles
        WHERE article_id IN ({placeholders})
    """
    results_df = pd.read_sql_query(query_sql, conn, params=result_ids)
    conn.close()
    
    # Add relevance scores
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
    
    # Apply filters
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
    
    search_cache[query_signature] = results_df
    return results_df

# Advanced contextual analysis
def contextual_query_understanding(query):
    """Understand user intent and extract key entities"""
    # Check cache first
    if "nlp_analysis" in st.session_state and st.session_state.nlp_analysis.get(query):
        return st.session_state.nlp_analysis[query]
    
    # Initialize entities
    entities = {
        "domains": [],
        "time_periods": [],
        "authors": [],
    }
    
    # Time period detection
    current_year = datetime.now().year
    time_patterns = [
        (r"since (\d{4})", lambda m: [int(m.group(1)), current_year]),
        (r"last (\d+) years", lambda m: [current_year - int(m.group(1)), current_year]),
        (r"from (\d{4}) to (\d{4})", lambda m: [int(m.group(1)), int(m.group(2))]),
        (r"(\d{4})-(\d{4})", lambda m: [int(m.group(1)), int(m.group(2))]),
        (r"in (\d{4})", lambda m: [int(m.group(1))]),
    ]
    
    for pattern, handler in time_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities["time_periods"] = handler(match)
            break
    
    # Author detection
    for author in authors:
        if author.lower() in query.lower():
            entities["authors"].append(author)
    
    # Domain detection
    if "cs." in query.lower():
        for token in query.split():
            if token.lower().startswith("cs."):
                entities["domains"].append(token)
    
    # Intent detection
    intent = "articles"
    intent_keywords = {
        "authors": ["researcher", "scientist", "professor", "expert", "author"],
        "trends": ["trend", "evolution", "growth", "over time", "publication pattern"],
        "collaborations": ["collaborator", "partner", "co-author", "work with", "network"],
        "topics": ["topic", "theme", "research area", "focus", "subject"],
        "articles": ["paper", "article", "publication", "study", "research"]
    }
    
    # Determine intent based on keywords
    for intent_type, keywords in intent_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            intent = intent_type
            break
    
    analysis = {
        "intent": intent,
        "entities": entities,
        "original_query": query
    }
    
    # Cache analysis
    if "nlp_analysis" not in st.session_state:
        st.session_state.nlp_analysis = {}
    st.session_state.nlp_analysis[query] = analysis
    
    return analysis

# Research functions
def get_top_authors(domain=None, year_range=None, min_papers=3, limit=10):
    """Get top authors with minimum paper threshold"""
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
            HAVING paper_count >= ?
            ORDER BY paper_count DESC
            LIMIT ?
        """
        params.extend([min_papers, limit])
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_publication_trends(domain=None, start_year=2020, end_year=None):
    """Get accurate publication trends"""
    if end_year is None:
        end_year = datetime.now().year
        
    conn = get_db_connection()
    try:
        # Create complete year range
        all_years = pd.DataFrame({'year': range(start_year, end_year + 1)})
        
        conditions = ["year BETWEEN ? AND ?"]
        params = [start_year, end_year]
        
        if domain:
            conditions.append("categories LIKE ?")
            params.append(f'%{domain}%')
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT year, COUNT(*) as paper_count
            FROM articles
            {where_clause}
            GROUP BY year
        """
        
        results = pd.read_sql_query(query, conn, params=params)
        
        # Merge with complete year range
        results = pd.merge(all_years, results, on='year', how='left').fillna(0)
        return results.sort_values('year')
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_collaborators(author_name, domain=None):
    """Get collaborators for an author"""
    conn = get_db_connection()
    try:
        # Find closest matching author name
        author_lower = author_name.lower()
        best_match = next((a for a in authors if author_lower in a.lower()), author_name)
        
        conditions = ["a1.author_name = ?", "a2.author_name != ?"]
        params = [best_match, best_match]
        
        if domain:
            conditions.append("a.categories LIKE ?")
            params.append(f'%{domain}%')
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT a2.author_name, COUNT(*) as collab_count
            FROM article_author aa1
            JOIN article_author aa2 ON aa1.article_id = aa2.article_id
            JOIN authors a1 ON aa1.author_id = a1.author_id
            JOIN authors a2 ON aa2.author_id = a2.author_id
            JOIN articles a ON aa1.article_id = a.article_id
            WHERE {where_clause}
            GROUP BY a2.author_name
            ORDER BY collab_count DESC
            LIMIT 10
        """
        
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def generate_topic_wordcloud(articles_df):
    """Generate word cloud of research topics"""
    if articles_df.empty:
        return None
        
    text = " ".join(articles_df['title'] + " " + articles_df['abstract'])
    if not text.strip():
        return None
        
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          stopwords=STOPWORDS, 
                          max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Generate intelligent responses
def generate_intelligent_response(analysis, results):
    """Generate contextual responses in natural language"""
    response = []
    visualizations = []
    query = analysis["original_query"]
    entities = analysis["entities"]
    
    # Custom response based on intent
    if analysis["intent"] == "authors":
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if results.empty:
            response.append("🔍 I couldn't find any prominent researchers matching your query.")
            response.append("💡 Try broadening your criteria or checking the spelling.")
        else:
            domain_info = f" in {domain}" if domain else ""
            response.append(f"👤 **Top researchers{domain_info}:**")
            
            # Display top authors
            for i, row in results.head(5).iterrows():
                response.append(f"- **{row['author_name']}** ({row['paper_count']} publications)")
            
            # Bar chart visualization
            if not results.empty:
                top_authors = results.head(10)
                fig = px.bar(
                    top_authors, 
                    x='author_name', 
                    y='paper_count',
                    labels={'author_name': 'Researcher', 'paper_count': 'Publications'},
                    title=f'Top Researchers{domain_info}',
                    color='paper_count'
                )
                fig.update_layout(height=400)
                visualizations.append(("Publication Count", fig))
    
    elif analysis["intent"] == "trends":
        # Determine time period
        time_periods = entities.get('time_periods', [])
        if not time_periods:
            start_year = 2020
            end_year = datetime.now().year
        elif len(time_periods) == 1:
            start_year = time_periods[0]
            end_year = datetime.now().year
        else:
            start_year = min(time_periods)
            end_year = max(time_periods)
        
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        results = get_publication_trends(domain=domain, start_year=start_year, end_year=end_year)
        
        if results.empty:
            response.append("📊 I couldn't identify any publication trends for your query.")
            response.append("💡 Try with a different domain or time period.")
        else:
            domain_info = f" in {domain}" if domain else ""
            response.append(f"📈 **Publication trends{domain_info} from {start_year} to {end_year}:**")
            
            # Line chart visualization
            if not results.empty:
                fig = px.line(
                    results, 
                    x='year', 
                    y='paper_count',
                    markers=True,
                    labels={'year': 'Year', 'paper_count': 'Publications'},
                    title=f'Publication Trends{domain_info}'
                )
                fig.update_layout(height=400)
                visualizations.append(("Publication Trends", fig))
    
    elif analysis["intent"] == "topics":
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if results.empty:
            response.append("🔍 I couldn't find any articles to analyze research topics.")
        else:
            domain_info = f" in {domain}" if domain else ""
            response.append(f"🌐 **Main research topics{domain_info}:**")
            
            # Word cloud visualization
            wordcloud_fig = generate_topic_wordcloud(results)
            if wordcloud_fig:
                visualizations.append(("Research Topics", wordcloud_fig))
    
    elif analysis["intent"] == "collaborations":
        author_name = entities.get("authors", [""])[0]
        
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        
        if not author_name:
            response.append("🔍 Please specify an author name to find collaborators.")
            response.append("💡 Example: 'Who collaborates with Yann LeCun?'")
        else:
            collaborators = get_collaborators(author_name, domain=domain)
            
            if collaborators.empty:
                response.append(f"🔍 I couldn't find collaborators for {author_name}.")
                response.append("💡 Try a different spelling or full name.")
            else:
                response.append(f"🤝 **{author_name}'s frequent collaborators:**")
                
                # List collaborators
                for i, row in collaborators.head(5).iterrows():
                    response.append(f"- **{row['author_name']}** ({row['collab_count']} joint papers)")
                
                # Bar chart visualization
                if not collaborators.empty:
                    fig = px.bar(
                        collaborators, 
                        x='author_name', 
                        y='collab_count',
                        labels={'author_name': 'Collaborator', 'collab_count': 'Joint Papers'},
                        title=f'Collaborators of {author_name}'
                    )
                    fig.update_layout(height=400)
                    visualizations.append(("Collaboration Network", fig))
    
    else:  # Default article search
        domain = entities.get("domains", [None])[0] if entities.get("domains") else None
        time_periods = entities.get('time_periods', [])
        
        if results.empty:
            response.append("🔍 I couldn't find any articles matching your search.")
            response.append("💡 Try broadening your search terms or adding synonyms.")
        else:
            domain_info = f" in {domain}" if domain else ""
            year_info = f" since {min(time_periods)}" if time_periods else ""
            
            response.append(f"📚 Found **{len(results)} relevant articles**{domain_info}{year_info}:")
            
            # Display top articles
            for i, row in results.head(3).iterrows():
                arxiv_link = f"https://arxiv.org/abs/{row['article_id']}"
                
                # Shorten long titles
                title = row['title']
                short_title = title if len(title) <= 80 else f"{title[:77]}..."
                
                response.append("---")
                response.append(f"### [{short_title}]({arxiv_link})")
                response.append(f"**Authors**: {row['authors']}")
                response.append(f"**Year**: {row['year']} | **Relevance**: {row['similarity_score']:.2f}")
                
                # DOI link if available
                if row['doi'] and row['doi'] != 'None':
                    doi_link = f"https://doi.org/{row['doi']}"
                    response.append(f"**DOI**: [Full paper]({doi_link})")
                
                # Abstract with expander
                if row['abstract']:
                    with st.expander("View abstract"):
                        st.write(textwrap.fill(row['abstract'], width=100))
            
            # Relevance score visualization
            if len(results) > 1:
                scores_df = results.head(10)[['title', 'similarity_score']]
                scores_df['title_short'] = scores_df['title'].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
                
                fig = px.bar(
                    scores_df, 
                    x='similarity_score', 
                    y='title_short',
                    orientation='h',
                    labels={'similarity_score': 'Relevance Score', 'title_short': 'Article'},
                    title='Most Relevant Articles',
                    color='similarity_score',
                    color_continuous_scale='Bluered'
                )
                fig.update_layout(height=500, showlegend=False)
                visualizations.append(("Article Relevance", fig))

    return response, visualizations

# User interface
st.title("💬 Papers Research Assistant")
st.caption("Research scientifique papers companion")

# Initialize conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.nlp_analysis = {}  # Cache for NLP analysis

# Sidebar for filters
with st.sidebar:
    st.header("⚙️ Filters")
    
    # Year filter
    min_year, max_year = 2020, datetime.now().year
    selected_years = st.slider(
        "Publication period", 
        min_value=min_year, 
        max_value=max_year, 
        value=(min_year, max_year)
    )
    
    # Domain filter
    try:
        conn = get_db_connection()
        categories = pd.read_sql_query("SELECT DISTINCT categories FROM articles", conn)['categories']
        conn.close()
        
        all_categories = set()
        for cat_list in categories:
            if cat_list:
                all_categories.update(cat_list.split())
        
        cs_categories = sorted([cat for cat in all_categories if cat.startswith('cs.')])
        
        selected_category = st.selectbox(
            "Research domain", 
            [""] + cs_categories
        )
    except:
        selected_category = None

# Display conversation history
for msg_idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🔬"):
            st.write(msg["content"])
            
            # Display visualizations
            if "visualizations" in msg:
                for viz_idx, (title, viz) in enumerate(msg["visualizations"]):
                    # Generate unique key for each visualization
                    unique_key = f"viz_{msg_idx}_{viz_idx}_{uuid.uuid4().hex[:8]}"
                    
                    if isinstance(viz, plt.Figure):
                        st.pyplot(viz)
                    elif hasattr(viz, 'show'):
                        st.plotly_chart(viz, use_container_width=True, key=unique_key)

# Handle user input
if prompt := st.chat_input("Ask about scientific research..."):
    # Add question to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing your query..."):
        start_time = time.time()
        
        # Advanced contextual analysis
        analysis = contextual_query_understanding(prompt)
        
        # Apply filters
        filters = {
            "year_range": selected_years,
            "domain": selected_category if selected_category else None,
        }
        
        # Handle different intents
        results = pd.DataFrame()
        visualizations = []
        
        try:
            if analysis["intent"] == "authors":
                results = get_top_authors(
                    domain=filters['domain'],
                    year_range=selected_years
                )
                
            elif analysis["intent"] == "trends":
                # Use extracted dates or default filters
                time_periods = analysis['entities'].get('time_periods', [])
                if not time_periods:
                    start_year = selected_years[0]
                    end_year = selected_years[1]
                elif len(time_periods) == 1:
                    start_year = time_periods[0]
                    end_year = selected_years[1]
                else:
                    start_year = min(time_periods)
                    end_year = max(time_periods)
                
                results = get_publication_trends(
                    domain=filters['domain'],
                    start_year=start_year,
                    end_year=end_year
                )
                
            elif analysis["intent"] == "topics":
                results = cached_semantic_search(analysis["original_query"], filters)
                
            elif analysis["intent"] == "collaborations":
                # Don't do semantic search for collaborators
                results = pd.DataFrame()
                
            else:  # Default article search
                results = cached_semantic_search(prompt, filters, k=100)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        
        # Generate response
        try:
            response_lines, visualizations = generate_intelligent_response(analysis, results)
            response_text = "\n".join(response_lines)
        except Exception as e:
            response_text = f"⚠️ Error generating response: {str(e)}"
            visualizations = []
        
        # Calculate response time
        response_time = time.time() - start_time
        response_text += f"\n\n⏱️ Response generated in {response_time:.2f} seconds"
        
        # Create assistant message
        assistant_msg = {
            "role": "assistant", 
            "content": response_text,
            "visualizations": visualizations
        }
        
        # Add to history
        st.session_state.chat_history.append(assistant_msg)
        
        # Reload interface
        st.rerun()



# Footer
st.divider()
st.caption("© 2025 Papers Research Assistant | Samia Regrai & Nouhaila Ennaouaoui")