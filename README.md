IMDB Sentiment Analysis Using RNN, LSTM & GRU
A deep learning project that classifies movie reviews as Positive or Negative using recurrent neural networks. The project compares Simple RNN, LSTM, and GRU models on the IMDB movie review dataset and provides an interactive Streamlit app for sentiment prediction.
Live Demo
Try the Streamlit Sentiment Analysis App
Note: This demo uses a temporary Cloudflare Tunnel from Google Colab. It works only while the Colab runtime, Streamlit server, and tunnel are running. The URL may change when the tunnel restarts.

Project Overview
- Task: Binary sentiment classification (Positive / Negative)
- Dataset: IMDB movie reviews (50,000 labeled reviews)
- Vocabulary size: 10,000 words
- Sequence length: 500 tokens (padding / truncation)
- Models: Simple RNN, LSTM, GRU
- Frameworks: TensorFlow / Keras, Scikit-learn, Streamlit
- Development environment: Google Colab
Dataset
The project uses the built-in IMDB dataset available in TensorFlow/Keras:
from tensorflow.keras.datasets import imdb

max_features = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
The standard split contains 25,000 training reviews and 25,000 test reviews. Labels are 0 for Negative and 1 for Positive. Keras supplies reviews as integer-encoded sequences, so a separate CSV download is not required.

Dataset source: https://ai.stanford.edu/~amaas/data/sentiment/
Project Workflow
1. Load the IMDB dataset with a 10,000-word vocabulary.
2. Pad or truncate each review to 500 tokens.
3. Build three models: Simple RNN, LSTM, and GRU.
4. Train each model using an Embedding layer, recurrent layer, Dropout, and a Sigmoid output.
5. Compare model performance on held-out data.
6. Save the trained GRU model as gru_sentiment_model.keras.
7. Load the saved model in a Streamlit app to predict sentiment from user-entered movie reviews.
Model Architecture
Each model follows a similar structure:
Input (500 token IDs)
       ↓
Embedding (10,000 vocabulary, 128 dimensions)
       ↓
SimpleRNN(64) / LSTM(64) / GRU(64)
       ↓
Dropout(0.3)
       ↓
Dense(1, activation='sigmoid')
       ↓
Positive / Negative Prediction
Training setup: Adam optimizer, binary cross-entropy loss, and accuracy metric. Early stopping can help prevent overfitting.
Evaluation
The models can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
Reported project result: GRU achieved 88.52% test accuracy in the project description. Results may differ when retraining with different settings or random seeds. Replace this figure with your actual Colab test result if it differs.
Run in Google Colab
1. Open Google Colab.
2. Upload your training notebook.
3. Optionally enable GPU: Runtime → Change runtime type → T4 GPU (if available).
4. Run the cells to load the dataset, preprocess reviews, and train RNN, LSTM, and GRU models.
5. Evaluate the models and save the GRU model:
gru_model.save('gru_sentiment_model.keras')
Run the Streamlit App in Colab
Install Streamlit:
!pip install -q streamlit
Create an app.py file in Colab (for example, with %%writefile app.py) that loads gru_sentiment_model.keras, preprocesses text, and displays predictions.

Start the app:
!streamlit run app.py --server.port 8501 > streamlit.log 2>&1 &
Expose the running app through a temporary Cloudflare Tunnel. The Live Demo link above points to the tunnel URL used for this project.
Important: Use the same vocabulary mapping, padding length, and text preprocessing conventions for inference as during training. Differences in preprocessing can reduce prediction quality.

Example Predictions
Movie review	Expected sentiment
"The movie was amazing and the acting was excellent."	Positive
"The story was boring and disappointing."	Negative


These are illustrative examples, not verified outputs from the trained model.
Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Streamlit
- Google Colab
- Cloudflare Tunnel (temporary demo access)
Future Improvements
- Deploy to a persistent hosting service instead of a temporary Colab tunnel.
- Save and reuse an identical text preprocessing pipeline for inference.
- Add training history plots and a side-by-side model comparison table.
- Improve handling of punctuation, unknown words, and unusually long reviews.
Author
A.S.ABHASH
