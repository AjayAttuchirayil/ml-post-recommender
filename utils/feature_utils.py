# utils/feature_utils.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def create_features():
    data = pd.read_csv('data/processed/user_post_interactions.csv')

    # TF-IDF of post text
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(data['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    data = pd.concat([data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Simulate user interests again (match with post topic)
    user_interests = {}
    for uid in data['user_id'].unique():
        user_interests[uid] = np.random.choice(['tech', 'food', 'travel', 'fitness', 'gaming', 'fashion', 'news'],
                                               size=np.random.randint(1, 4), replace=False).tolist()

    data['topic_match'] = data.apply(lambda row: int(row['topic'] in user_interests[row['user_id']]), axis=1)

    # One-hot for interaction type
    ohe = pd.get_dummies(data['interaction'], prefix='interact')
    data = pd.concat([data, ohe], axis=1)

    # Select final features
    features = data.drop(columns=['interaction', 'text', 'post_id', 'user_id', 'topic'])  # 👈 added 'topic'
    labels = data['label']

    features.to_csv('data/processed/features.csv', index=False)
    labels.to_csv('data/processed/labels.csv', index=False)
    print("✅ Features and labels saved in data/processed/")

if __name__ == "__main__":
    create_features()
