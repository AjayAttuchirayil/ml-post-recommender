# 🧠 Personalized Post Recommender

A mini machine learning project inspired by Meta’s recommendation systems. The goal is to simulate a content feed that recommends posts to users based on their interests and past behavior.

---

## 📌 Problem

Given a set of users, posts, and user-post interactions (likes/views), train a model that predicts how likely a user is to like a new post.

---

## 📦 Dataset

Simulated data using:
- 100 users
- 1,000 posts (across 7 topics)
- Synthetic interactions (like, view) based on user-topic match

---

## 🔧 ML Pipeline

| Step                | Details |
|---------------------|---------|
| **Data simulation** | Generate user interests and synthetic post text |
| **Feature engineering** | TF-IDF embeddings of post text, topic match, interaction one-hot |
| **Model** | LightGBM classifier |
| **Evaluation** | ROC-AUC, can extend to NDCG@k / MAP@k for ranking |
| **App** | Streamlit demo to simulate recommendations |

---

## 🧠 Features Used

- TF-IDF features from post text (top 100 tokens)
- Binary flag if post topic matches user interests
- One-hot encoding for interaction type

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/personalized-post-recommender.git
cd personalized-post-recommender
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python run_pipeline.py
