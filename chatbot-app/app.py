from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import spacy
import re
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained LSTM model
print("Loading LSTM model...")
lstm_model = tf.keras.models.load_model('lstm_model.h5')
print("LSTM model loaded successfully.")

# Load JSON data into a DataFrame
with open('train_df.json', 'r') as file:
    data = [json.loads(line) for line in file]
    data_df = pd.DataFrame(data)  # Convert JSON to DataFrame
    print("JSON data loaded successfully.")

# Initialize spaCy for entity extraction
nlp = spacy.load("en_core_web_sm")

# Load tokenizer used during training
with open('tokenizer.json', 'r') as tokenizer_file:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    tokenizer = tokenizer_from_json(json.load(tokenizer_file))

# Maximum sequence length used during LSTM training
max_sequence_length = 100  # Adjust to match the training setup

# Function to preprocess text for LSTM model
def preprocess_text(text, tokenizer, max_len):
    """Tokenize and pad the input text."""
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    return padded_seq

# Function to find the closest matching text in the JSON file
def find_closest_match(user_query, data_df):
    """Find the closest matching text from the JSON data."""
    user_query_doc = nlp(user_query)
    max_similarity = 0
    best_match = None

    for _, row in data_df.iterrows():
        doc = nlp(row['text'])
        similarity = user_query_doc.similarity(doc)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = row

    return best_match

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()

    # Preprocess the user query
    processed_input = preprocess_text(user_message, tokenizer, max_sequence_length)

    # Get LSTM prediction
    prediction = lstm_model.predict(processed_input)
    predicted_value = prediction[0][0]  # Adjust based on your model's output

    # Find the closest match in the JSON file
    best_match = find_closest_match(user_message, data_df)

    if best_match is not None:
        response_text = best_match['text']
        actual_value = best_match['value']
        return jsonify({
            'response': f"Based on your query, the closest match is '{response_text}' with an actual value of {actual_value}. Predicted value is {predicted_value:.2f}."
        })
    else:
        return jsonify({'response': "I'm sorry, I couldn't find a relevant match for your query."})

if __name__ == '__main__':
    app.run(debug=True)
