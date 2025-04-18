# app/streamlit_app.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import os

# Load processed data
@st.cache_data
def load_data():
    try:
        features = pd.read_csv("data/processed/features.csv")
        raw_data = pd.read_csv("data/processed/user_post_interactions.csv")
        return features, raw_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_model():
    try:
        return lgb.Booster(model_file="models/lgb_model.txt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load everything
features, raw_data = load_data()
model = load_model()

st.title("📬 Personalized Post Recommender")

if features is None or raw_data is None or model is None:
    st.stop()

# Select user
user_ids = raw_data['user_id'].unique()
selected_user = st.selectbox("Choose a user", sorted(user_ids))

# Filter data for selected user
user_data = raw_data[raw_data['user_id'] == selected_user].copy()
user_features = features.loc[user_data.index].copy()

# Predict probabilities
user_data['score'] = model.predict(user_features)

# Top N recommendations
top_n = st.slider("How many posts to show?", 5, 20, 10)
top_posts = user_data.sort_values(by='score', ascending=False).head(top_n)

st.markdown("### 🔮 Recommended Posts")
for _, row in top_posts.iterrows():
    st.markdown(f"**Topic:** `{row['topic']}`")
    st.markdown(f"📝 {row['text']}")
    st.progress(float(row['score']))
    st.markdown("---")
