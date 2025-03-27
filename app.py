import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from recommender import get_recommendation
from huggingface_hub import InferenceClient
import os

st.title("🧠 SuppleSense – AI Customer Insights & Retention Tool")

# Load data and models
df = pd.read_csv("data/segmented_customers.csv")
model = joblib.load("churn_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.write("## Customer Overview")
st.dataframe(df.head())

# Churn distribution chart
st.write("### 📊 Churn Probability Distribution")
fig, ax = plt.subplots()
churn_probs = []
for _, row in df.iterrows():
    gender_val = 0 if row['gender'] == 'M' else 1
    features = [[
        row['age'], gender_val, row['purchase_count'],
        row['avg_order_value'], row['days_since_last_purchase'], row['email_open_rate']
    ]]
    prob = model.predict_proba(features)[0][1]
    churn_probs.append(prob)

ax.hist(churn_probs, bins=10, color='skyblue', edgecolor='black')
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Number of Customers")
ax.set_title("Churn Risk Distribution")
st.pyplot(fig)

# Select customer
st.write("---")
customer_id = st.selectbox("Select Customer ID", df['customer_id'])
row = df[df['customer_id'] == customer_id].iloc[0]

# Ensure gender is encoded (M=0, F=1)
gender_val = 0 if row['gender'] == 'M' else 1

# Prepare input features
input_features = [[
    row['age'], gender_val, row['purchase_count'],
    row['avg_order_value'], row['days_since_last_purchase'], row['email_open_rate']
]]

# Predict churn
churn_proba = model.predict_proba(input_features)[0][1]
segment = row['segment']

# Show results
st.write(f"**Churn Probability:** {churn_proba:.2f}")
st.write(f"**Segment ID:** {segment}")
st.write(f"**Recommended Action:** {get_recommendation(segment)}")

# Email Generation Assistant
st.write("---")
st.subheader("✉️ Generate Re-Engagement Email")

if churn_proba > 0.5:
    if st.button("Generate Email with Hugging Face"):
        prompt = f"""
        Write a short and friendly marketing email to re-engage a {row['age']}-year-old {'woman' if row['gender'] == 'F' else 'man'} 
        who hasn’t purchased in {row['days_since_last_purchase']} days. 
        Mention our health supplements and offer a loyalty discount.
        """
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        client = InferenceClient(token=hf_token)
        response = client.text_generation(prompt=prompt, model="tiiuae/falcon-7b-instruct", max_new_tokens=300)
        st.success("Here’s your email 👇")
        st.write(response.strip())
