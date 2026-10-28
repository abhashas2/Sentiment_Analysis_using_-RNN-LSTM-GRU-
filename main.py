import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# ==========================================
# Step 1: Load IMDB word index and model
# ==========================================

max_features = 10000
max_len = 500

word_index = imdb.get_word_index()

reverse_word_index = {
    value: key for key, value in word_index.items()
}

model = load_model("Simple_rnn_imdb (1).h5")


# ==========================================
# Step 2: Helper Functions
# ==========================================

def decode_review(encoded_review):
    return ' '.join(
        reverse_word_index.get(i - 3, '?')
        for i in encoded_review
    )


def preprocess_text(text):

    words = text.lower().split()

    encoded_review = []

    for word in words:

        word_id = word_index.get(word)

        if word_id is None or word_id + 3 >= max_features:
            encoded_review.append(2)

        else:
            encoded_review.append(word_id + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=max_len
    )

    return padded_review


# ==========================================
# Step 3: Prediction Function
# ==========================================

def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    sentiment = (
        'Positive'
        if prediction[0][0] > 0.5
        else 'Negative'
    )

    return sentiment, prediction[0][0]


# ==========================================
# Step 4: Streamlit App
# ==========================================

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as positive or negative."
)

user_input = st.text_area("Movie Review")


if st.button("Classify"):

    if user_input.strip():

        sentiment, score = predict_sentiment(user_input)

        st.write(f"Sentiment: **{sentiment}**")

        st.write(
            f"Prediction Score: **{score:.4f}**"
        )

    else:

        st.write("Please enter a movie review.")