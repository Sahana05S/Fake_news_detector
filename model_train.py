import pandas as pd
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_isot_data():
    print("🔹 Loading ISOT dataset...")
    fake_df = pd.read_csv("data/ISOT/Fake.csv")
    real_df = pd.read_csv("data/ISOT/True.csv")

    fake_df["label"] = 0
    real_df["label"] = 1

    df = pd.concat([fake_df, real_df])
    df = df.sample(frac=1).reset_index(drop=True)
    return df["text"], df["label"]

def load_liar_data():
    print("🔹 Loading LIAR dataset...")
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
               'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
               'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts',
               'context']

    # Load all three splits and concatenate
    train_df = pd.read_csv("data/LIAR/train.tsv", sep='\t', names=columns)
    valid_df = pd.read_csv("data/LIAR/valid.tsv", sep='\t', names=columns)
    test_df  = pd.read_csv("data/LIAR/test.tsv", sep='\t', names=columns)

    df = pd.concat([train_df, valid_df, test_df])

    # Map labels to binary: fake (0), real (1)
    fake_labels = ['false', 'pants-fire', 'barely-true']
    real_labels = ['true', 'mostly-true', 'half-true']

    df = df[df['label'].isin(fake_labels + real_labels)].copy()
    df['label'] = df['label'].apply(lambda x: 0 if x in fake_labels else 1)

    return df["statement"], df["label"]

def train_and_save_model(X, y, output_path):
    print("🔹 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔹 Training model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"✅ Model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["isot", "liar"], required=True, help="Which dataset to train on.")
    args = parser.parse_args()

    if args.dataset == "isot":
        X, y = load_isot_data()
        output_model_path = "model/isot_model.pkl"
    elif args.dataset == "liar":
        X, y = load_liar_data()
        output_model_path = "model/liar_model.pkl"
    else:
        raise ValueError("Unknown dataset selected.")

    train_and_save_model(X, y, output_model_path)

if __name__ == "__main__":
    main()
