# data_handler.py
import os
import pandas as pd
import numpy as np
import re, string, zipfile, urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def download_dataset(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    file_path = os.path.join(data_dir, "SMSSpamCollection")
    if not os.path.exists(file_path):
        print("[INFO] Downloading SMS Spam Collection dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("[INFO] Dataset downloaded and extracted.")
    return file_path

def _stratified_client_indices(y, n_clients):
    # keep spam ratio similar per client
    y = np.asarray(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)
    splits = [[] for _ in range(n_clients)]
    for i, ix in enumerate(idx_pos):
        splits[i % n_clients].append(ix)
    for i, ix in enumerate(idx_neg):
        splits[i % n_clients].append(ix)
    return [np.array(sorted(s)) for s in splits]

def load_and_split_data(filepath=None, n_clients=5, test_size=0.2, val_size=0.1):
    if filepath is None:
        filepath = download_dataset("data")

    df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["cleaned"] = df["message"].apply(clean_text)

    # Train/test split first (stratified)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )

    # Vectorizer fit ONLY on train to avoid leakage
    vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
    X_train = vectorizer.fit_transform(train_df["cleaned"])
    y_train = train_df["label"].to_numpy()

    X_test = vectorizer.transform(test_df["cleaned"])
    y_test = test_df["label"].to_numpy()

    # carve a small validation set from train for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )

    # Stratified client partition from remaining training set
    client_idx = _stratified_client_indices(y_tr, n_clients)
    clients = [(X_tr[ix], y_tr[ix]) for ix in client_idx]

    return {
        "clients": clients,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "vectorizer": vectorizer,
    }
