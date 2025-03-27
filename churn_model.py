import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_churn_model():
    df = pd.read_csv("data/customer_data.csv")

    # Encode gender to numeric: M = 0, F = 1
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

    # Define features/target
    X = df.drop(columns=['customer_id', 'churned'])
    y = df['churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("✅ Churn Model Accuracy:", round(acc, 3))

    # Save model
    joblib.dump(model, 'churn_model.pkl')
    print("✅ churn_model.pkl saved to:", os.path.abspath("churn_model.pkl"))

if __name__ == "__main__":
    train_churn_model()
