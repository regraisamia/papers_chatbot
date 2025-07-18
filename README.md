
# 📚 Papers Research Assistant  
![Papers Research Assistant]

**Papers Research Assistant** est une application intelligente qui vous permet d'explorer et d'analyser les dernières publications scientifiques en informatique.  
Basée sur les archives **arXiv**, elle utilise l’intelligence artificielle pour comprendre vos requêtes et fournir des réponses contextuelles enrichies de visualisations interactives.

---

##  Fonctionnalités clés

-  **Recherche sémantique** — Trouvez des articles pertinents en posant des questions en langage naturel  
-  **Analyse de tendances** — Suivez l’évolution des publications par domaine ou par période  
-  **Réseaux de collaboration** — Identifiez les auteurs qui travaillent ensemble  
-  **Visualisations interactives** — Explorez les résultats à l’aide de graphiques dynamiques  
-  **Réponses instantanées** — Obtenez des résultats riches en moins d’une seconde

---

##  Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-00B0FF)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.7.0-4DC71F)
![SQLite](https://img.shields.io/badge/SQLite-3.42.0-003B57)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75)

---

##  Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/papers-research-assistant.git
cd papers-research-assistant
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate     # Pour Linux/Mac
venv\Scripts\activate        # Pour Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

##  Configuration des données

### 1. Télécharger les métadonnées arXiv

- Récupérez le fichier `arxiv-metadata-oai-snapshot.json` depuis [Kaggle](https://www.kaggle.com/datasets)
- Placez-le dans le dossier `arxiv/data/` du projet

### 2. Préparer les fichiers nettoyés

```bash
jupyter notebook data-prep.ipynb
```

### 3. Créer la base relationnelle SQLite

```bash
jupyter notebook stockage.ipynb
```

### 4. Générer l’index vectoriel FAISS

```bash
jupyter notebook indexation.ipynb
```

---

## 🚀 Lancer l'application

```bash
streamlit run app.py
```

---

## 💬 Exemples de requêtes

- `recent articles about computer vision`
- `articles about NLP in medical fields`
- `AI trends from 2020 to 2024`
- `collaborators with Jennifer Doherty`
- `top authors in deep learning`
- `how many papers were published in quantum computing in 2023`

---

## 🛠️ Personnalisation

Dans `app.py`, vous pouvez modifier :

### 📅 Période de publication par défaut :

```python
selected_years = st.slider("Période de publication", 2020, 2025, (2020, 2024))
```

### 🤖 Modèle d’embedding sémantique :

```python
model = SentenceTransformer("all-mpnet-base-v2")
```

Vous pouvez aussi tester d'autres modèles comme `all-MiniLM-L6-v2` pour un traitement plus rapide.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour participer :

1. Forkez ce dépôt
2. Créez une nouvelle branche :
   ```bash
   git checkout -b feature/ma-fonction
   ```
3. Faites vos modifications et commit :
   ```bash
   git commit -m "Ajout de la fonctionnalité X"
   ```
4. Pushez vos modifications :
   ```bash
   git push origin feature/ma-fonction
   ```
5. Créez une Pull Request 📩

---

## 📢 À propos

Projet réalisé dans le cadre du Master Business Intelligence et Big Data Analytiques (BIBDA).  
Un assistant scientifique intelligent pour l’analyse automatisée des publications arXiv.

**Papers Research Assistant © 2025**  
Développé par : [Samia Regragui](https://github.com/regraisamia) & [Nouhaila Ennaouaoui](#)
