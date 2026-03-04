# 📚 Papers Research Assistant

**Papers Research Assistant** is an intelligent application designed to explore and analyze the latest scientific publications in computer science. 
Built on top of the **arXiv** archives, it leverages Artificial Intelligence to understand natural language queries and provide contextual answers enriched with interactive visualizations.

---

## ✨ Key Features

- **Semantic Search:** Find highly relevant papers by asking questions in natural language.
- **Trend Analysis:** Track the evolution of publications by specific domains or time periods.
- **Collaboration Networks:** Identify authors who frequently co-author and work together.
- **Interactive Visualizations:** Explore query results through dynamic, interactive graphs.
- **Instant Answers:** Retrieve rich, contextualized results in under a second.

---

## 💻 Technologies Used

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-00B0FF)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.7.0-4DC71F)
![SQLite](https://img.shields.io/badge/SQLite-3.42.0-003B57)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/papers-research-assistant.git
cd papers-research-assistant
```

### 2. Create a virtual environment
```bash
# For Linux/Mac
python -m venv venv
source venv/bin/activate  

# For Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🗄️ Data Configuration

### 1. Download arXiv Metadata

- Download the `arxiv-metadata-oai-snapshot.json` file from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Place it inside the `arxiv/data/` directory of the project

### 2. Prepare and Clean Data
```bash
jupyter notebook data-prep.ipynb
```

### 3. Create the SQLite Relational Database
```bash
jupyter notebook stockage.ipynb
```

### 4. Generate the FAISS Vector Index
```bash
jupyter notebook indexation.ipynb
```

---

## 🏃♂️ Running the Application

```bash
streamlit run app.py
```

---

## 💬 Example Queries

- recent articles about computer vision
- articles about NLP in medical fields
- AI trends from 2020 to 2024
- collaborators with Jennifer Doherty
- top authors in deep learning
- how many papers were published in quantum computing in 2023

---

## 🛠️ Customization

Inside `app.py`, you can modify the following parameters:

### 📅 Default Publication Period:
```python
selected_years = st.slider("Publication Period", 2020, 2025, (2020, 2024))
```

### 🤖 Semantic Embedding Model:
```python
model = SentenceTransformer("all-mpnet-base-v2")
```

**Note:** You can test other models like `all-MiniLM-L6-v2` for faster processing times.

---

## 🤝 Contribution

Contributions are welcome! To contribute:

1. Fork this repository.

2. Create a new branch:
```bash
git checkout -b feature/my-feature
```

3. Commit your changes:
```bash
git commit -m "Add feature X"
```

4. Push to the branch:
```bash
git push origin feature/my-feature
```

5. Open a Pull Request 📩

---

## 🎓 About

This project was developed as part of the **Master's in Business Intelligence and Big Data Analytics (BIBDA)** program.
It serves as an intelligent scientific assistant for the automated analysis of arXiv publications.

---

**Papers Research Assistant** © 2025  
Developed by: **Samia Regrai** & **Nouhaila Ennaouaoui**
