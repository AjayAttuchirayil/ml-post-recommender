# utils/data_loader.py

import pandas as pd
import numpy as np
import random
from faker import Faker
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

def simulate_data():
    NUM_USERS = 100
    NUM_POSTS = 1000
    user_ids = [f"user_{i}" for i in range(NUM_USERS)]
    topics = ['tech', 'food', 'travel', 'fitness', 'gaming', 'fashion', 'news']

    # Posts
    posts = []
    for i in range(NUM_POSTS):
        post_id = f"post_{i}"
        topic = random.choice(topics)
        text = fake.sentence(nb_words=12)
        posts.append((post_id, topic, text))
    posts_df = pd.DataFrame(posts, columns=['post_id', 'topic', 'text'])

    # User interests
    user_interests = {user: random.sample(topics, k=random.randint(1, 3)) for user in user_ids}

    # Interactions
    interactions = []
    for user in user_ids:
        for _, row in posts_df.iterrows():
            topic_match = row['topic'] in user_interests[user]
            if topic_match and random.random() < 0.25:
                interactions.append((user, row['post_id'], 'like'))
            elif random.random() < 0.05:
                interactions.append((user, row['post_id'], 'view'))

    interactions_df = pd.DataFrame(interactions, columns=['user_id', 'post_id', 'interaction'])
    data = interactions_df.merge(posts_df, on='post_id', how='left')
    data['label'] = data['interaction'].apply(lambda x: 1 if x == 'like' else 0)

    os.makedirs("data/processed", exist_ok=True)
    data.to_csv('data/processed/user_post_interactions.csv', index=False)
    print("✅ Data saved to data/processed/user_post_interactions.csv")

if __name__ == "__main__":
    simulate_data()
