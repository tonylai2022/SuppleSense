import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def segment_customers():
    df = pd.read_csv("data/customer_data.csv")
    features = df[['age', 'purchase_count', 'avg_order_value', 'email_open_rate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['segment'] = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, 'kmeans_model.pkl')
    print("✅ kmeans_model.pkl saved")

    output_path = "data/segmented_customers.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Segmented data saved to {output_path}")

# Run it
if __name__ == "__main__":
    segment_customers()
