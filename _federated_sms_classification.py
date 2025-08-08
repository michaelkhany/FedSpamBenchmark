
import os
import re
import string
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import urllib.request
import zipfile

# 1. Load the Dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = "smsspamcollection.zip"
DATA_PATH = "SMSSpamCollection"

# Download if not exists
if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall()
    print("Download and extraction complete.")


df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# 2. Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['cleaned'] = df['message'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].values

# 3. Centralized Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
central_model = LogisticRegression(max_iter=200)
central_model.fit(X_train, y_train)
y_pred = central_model.predict(X_test)

central_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}
print("Centralized Model Metrics:", central_metrics)

# Save centralized model and results
joblib.dump(central_model, "central_model.joblib")
with open("central_metrics.json", "w") as f:
    json.dump(central_metrics, f, indent=4)


# 4. Simulate Federated Learning
n_clients = 5
client_data_size = len(df) // n_clients
clients = []

indices = np.arange(len(df))
np.random.shuffle(indices)

for i in range(n_clients):
    start = i * client_data_size
    end = (i + 1) * client_data_size if i != n_clients - 1 else len(df)
    client_indices = indices[start:end]
    client_X = X[client_indices]
    client_y = y[client_indices]
    clients.append((client_X, client_y))

# Train local models and average coefficients
coef_sum = np.zeros(X.shape[1])
intercept_sum = 0

for i, (X_local, y_local) in enumerate(clients):
    model = LogisticRegression(max_iter=200)
    model.fit(X_local, y_local)
    coef_sum += model.coef_.flatten()
    intercept_sum += model.intercept_[0]

# Aggregate (FedAvg)
fed_coef = coef_sum / n_clients
fed_intercept = intercept_sum / n_clients

# Evaluate the federated model
fed_model = LogisticRegression()
fed_model.coef_ = np.array([fed_coef])
fed_model.intercept_ = np.array([fed_intercept])
fed_model.classes_ = np.array([0, 1])  # Must manually set this for sklearn models

y_fed_pred = fed_model.predict(X_test)

federated_metrics = {
    "accuracy": accuracy_score(y_test, y_fed_pred),
    "precision": precision_score(y_test, y_fed_pred),
    "recall": recall_score(y_test, y_fed_pred),
    "f1_score": f1_score(y_test, y_fed_pred)
}
print("Federated Model Metrics:", federated_metrics)

# Save federated results and model
joblib.dump(fed_model, "federated_model.joblib")
with open("federated_metrics.json", "w") as f:
    json.dump(federated_metrics, f, indent=4)
