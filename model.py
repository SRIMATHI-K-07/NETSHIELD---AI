import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# EXPANDED SYNTHETIC DATASET (simulates real-world patterns)
# In a real project, replace this with the PhiUSIIL / UCI Phishing Dataset
# Download from: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
# ─────────────────────────────────────────────

def generate_dataset():
    np.random.seed(42)
    n = 300

    # Safe URLs: short, few dots, https, old domain, no suspicious keywords
    safe = {
        'URL_Length':     np.random.randint(10, 40, n // 2),
        'Num_Dots':       np.random.randint(1, 3,  n // 2),
        'HTTPS':          np.ones(n // 2, dtype=int),
        'Domain_Age':     np.random.uniform(2, 15, n // 2),
        'Num_Hyphens':    np.random.randint(0, 2,  n // 2),
        'Num_Slashes':    np.random.randint(1, 4,  n // 2),
        'Has_IP':         np.zeros(n // 2, dtype=int),
        'Suspicious_Keywords': np.zeros(n // 2, dtype=int),
        'Subdomain_Level':np.random.randint(0, 2,  n // 2),
        'Has_At_Symbol':  np.zeros(n // 2, dtype=int),
        'Result':         np.zeros(n // 2, dtype=int),
    }

    # Phishing URLs: long, many dots, no https, new domain, suspicious keywords
    phish = {
        'URL_Length':     np.random.randint(40, 120, n // 2),
        'Num_Dots':       np.random.randint(3, 8,   n // 2),
        'HTTPS':          np.random.randint(0, 2,   n // 2),
        'Domain_Age':     np.random.uniform(0, 1,   n // 2),
        'Num_Hyphens':    np.random.randint(1, 6,   n // 2),
        'Num_Slashes':    np.random.randint(3, 10,  n // 2),
        'Has_IP':         np.random.randint(0, 2,   n // 2),
        'Suspicious_Keywords': np.random.randint(0, 3, n // 2),
        'Subdomain_Level':np.random.randint(2, 5,   n // 2),
        'Has_At_Symbol':  np.random.randint(0, 2,   n // 2),
        'Result':         np.ones(n // 2, dtype=int),
    }

    df_safe  = pd.DataFrame(safe)
    df_phish = pd.DataFrame(phish)
    df = pd.concat([df_safe, df_phish], ignore_index=True).sample(frac=1, random_state=42)
    return df


FEATURE_COLS = [
    'URL_Length', 'Num_Dots', 'HTTPS', 'Domain_Age',
    'Num_Hyphens', 'Num_Slashes', 'Has_IP',
    'Suspicious_Keywords', 'Subdomain_Level', 'Has_At_Symbol'
]


def train_and_save():
    df = generate_dataset()
    X = df[FEATURE_COLS]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Ensemble: Random Forest + Gradient Boosting + Logistic Regression ──
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    lr  = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    y_pred   = ensemble.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    cv_scores = cross_val_score(ensemble, X, y, cv=5)
    print(f"Test Accuracy  : {accuracy}%")
    print(f"CV Accuracy    : {round(cv_scores.mean() * 100, 2)}% ± {round(cv_scores.std() * 100, 2)}%")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': ensemble, 'accuracy': accuracy}, f)

    print("✅ Model saved to model.pkl")
    return ensemble, accuracy


if __name__ == '__main__':
    train_and_save()
