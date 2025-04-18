# utils/train_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model():
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/labels.csv').squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"🎯 ROC-AUC on test set: {auc:.4f}")

    model.booster_.save_model('models/lgb_model.txt')
    print("✅ Model saved to models/lgb_model.txt")

if __name__ == "__main__":
    train_model()
