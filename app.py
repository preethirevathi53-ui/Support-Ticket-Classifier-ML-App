import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    'text': [
        'Payment failed again',
        'Transaction error',
        'App crashing',
        'Login issue',
        'Forgot password',
        'Order not delivered'
    ],
    'category': [
        'Payment','Payment','Technical','Account','Account','Delivery'
    ]
}

df = pd.DataFrame(data)

# Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

model = MultinomialNB()
model.fit(X, y)

# Priority logic
def get_priority(text):
    text = text.lower()
    if "failed" in text or "error" in text or "crash" in text:
        return "High"
    elif "not" in text or "delay" in text or "forgot" in text:
        return "Medium"
    else:
        return "Low"

# UI
st.title("Support Ticket Classifier")

user_input = st.text_input("Enter your issue:")

if st.button("Predict"):
    vec = vectorizer.transform([user_input])
    category = model.predict(vec)[0]
    priority = get_priority(user_input)

    st.success(f"Category: {category}")
    st.warning(f"Priority: {priority}")