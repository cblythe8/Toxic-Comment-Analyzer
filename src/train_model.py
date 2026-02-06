"""
Train a toxic comment classifier and save predictions for error analysis.
"""
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Configuration
SAMPLE_SIZE = 50000  # Use 50k samples for manageable training
TOXICITY_THRESHOLD = 0.5
RANDOM_STATE = 42

def load_data():
    """Load and prepare the Civil Comments dataset."""
    print("Loading Civil Comments dataset...")
    dataset = load_dataset("google/civil_comments", split=f"train[:{SAMPLE_SIZE}]")
    df = pd.DataFrame(dataset)

    # Create binary label
    df['is_toxic'] = (df['toxicity'] >= TOXICITY_THRESHOLD).astype(int)

    print(f"Loaded {len(df)} samples")
    print(f"Toxic: {df['is_toxic'].sum()} ({df['is_toxic'].mean()*100:.1f}%)")
    print(f"Non-toxic: {(1-df['is_toxic']).sum()} ({(1-df['is_toxic'].mean())*100:.1f}%)")

    return df

def train_model(df):
    """Train TF-IDF + Logistic Regression model."""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['is_toxic'],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df['is_toxic']
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,            # Ignore rare terms
        max_df=0.95          # Ignore very common terms
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle imbalanced data
        random_state=RANDOM_STATE
    )
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]  # Probability of toxic

    return model, vectorizer, X_test, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred):
    """Print model evaluation metrics."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-toxic', 'Toxic']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Non-toxic  Toxic")
    print(f"Actual Non-toxic  {cm[0,0]:>8}  {cm[0,1]:>5}")
    print(f"Actual Toxic      {cm[1,0]:>8}  {cm[1,1]:>5}")

    # Error counts
    fp = cm[0, 1]  # False positives (non-toxic predicted as toxic)
    fn = cm[1, 0]  # False negatives (toxic predicted as non-toxic)
    print(f"\nFalse Positives (over-censorship): {fp}")
    print(f"False Negatives (missed toxicity): {fn}")

    return cm

def save_results(model, vectorizer, X_test, y_test, y_pred, y_pred_proba, df):
    """Save model and predictions for dashboard."""
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    with open(os.path.join(data_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(data_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Create results dataframe for error analysis
    print("Saving predictions for error analysis...")
    results_df = pd.DataFrame({
        'text': X_test.values,
        'actual': y_test.values,
        'predicted': y_pred,
        'confidence': y_pred_proba,
        'correct': (y_test.values == y_pred).astype(int)
    })

    # Add original metadata (toxicity scores, subcategories)
    test_indices = X_test.index
    for col in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']:
        results_df[col] = df.loc[test_indices, col].values

    # Add text length
    results_df['text_length'] = results_df['text'].str.len()

    results_df.to_csv(os.path.join(data_dir, 'predictions.csv'), index=False)
    print(f"Saved predictions to {os.path.join(data_dir, 'predictions.csv')}")

    return results_df

def main():
    # Load data
    df = load_data()

    # Train model
    model, vectorizer, X_test, y_test, y_pred, y_pred_proba = train_model(df)

    # Evaluate
    evaluate_model(y_test, y_pred)

    # Save for dashboard
    results_df = save_results(model, vectorizer, X_test, y_test, y_pred, y_pred_proba, df)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Next step: Run the Streamlit dashboard to explore errors")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
