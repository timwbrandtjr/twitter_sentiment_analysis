import pandas as pd
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Part 1: Load and preprocess the dataset
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv("/Users/n/Downloads/archive/training.1600000.processed.noemoticon.csv", encoding='ISO-8859-1', header=None)
df.columns = columns
df = df[['target', 'text']]
df['text'] = df['text'].str.replace('@', '')  # Remove mentions
df['text'] = df['text'].str.replace('[^a-zA-Z]', ' ')  # Keep only letters

# Part 2: Train the sentiment analysis model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['target']
model = LogisticRegression()
model.fit(X, y)

# Part 3: Function to predict sentiment
def predict_sentiment(sentence, model, vectorizer):
    sentence = re.sub('@', '', sentence)  # Remove mentions
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)  # Keep only letters
    sentence_vectorized = vectorizer.transform([sentence])
    predicted_score = model.predict(sentence_vectorized)[0]

    if predicted_score == 4:
        return "positive"
    elif predicted_score == 0:
        return "negative"
    else:
        return "unexpected"

# Streamlit App
def main():
    st.title("Sentiment Analysis App")

    st.markdown("""
        This app uses a logistic regression model to predict the sentiment of a given text.
        Enter a sentence, and it will predict whether it is positive or negative.
    """)

    # User input
    new_sentence = st.text_input("Enter a new sentence:")

    if new_sentence:
        # Prediction
        sentiment_prediction = predict_sentiment(new_sentence, model, vectorizer)

        st.subheader("Prediction:")
        if sentiment_prediction == "positive":
            st.success("The model predicts the sentence is positive.")
        elif sentiment_prediction == "negative":
            st.error("The model predicts the sentence is negative.")
        else:
            st.warning("Unexpected prediction score.")

if __name__ == "__main__":
    main()