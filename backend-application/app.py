import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from duckling import DucklingWrapper
import joblib
import re

# Load models and vectorizers
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')  # TF-IDF vectorizer
regression_model = joblib.load('models/xgboost_model.joblib')  # Regression model
scaler_model = joblib.load('models/scaler.joblib')  # Scaler for inverse-transform

# Load Hugging Face models
bert_intent_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
dialo_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
dialo_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Initialize Duckling for entity extraction
duckling = DucklingWrapper()

# Flask App
app = Flask(__name__)

# Intent Recognition
def recognize_intent(user_input):
    """
    Recognize intent using Hugging Face BERT pipeline.
    """
    intent_result = bert_intent_pipeline(user_input)
    if intent_result[0]["label"] == "LABEL_1":  # Adjust labels as per your dataset
        return "predict_revenue"
    elif "hello" in user_input.lower() or "hi" in user_input.lower():
        return "greet"
    elif "bye" in user_input.lower() or "exit" in user_input.lower():
        return "exit"
    return "fallback"

# Enhanced Entity Extraction
def extract_entities(user_input):
    """
    Extract entities like quarter and year using Duckling and regex as fallback.
    """
    try:
        parsed = duckling.parse_time(user_input)
        quarter, year = None, None

        # Extract entities from Duckling
        for entity in parsed:
            if "value" in entity and "grain" in entity["value"]:
                if entity["value"]["grain"] == "quarter":
                    quarter = int(entity["value"]["value"][-1])  # Extract Q1, Q2, etc.
                if entity["value"]["grain"] == "year":
                    year = int(entity["value"]["value"][:4])  # Extract 2024, etc.

        # Fallback for quarter and year
        if quarter is None:
            match_quarter = re.search(r'\bQ([1-4])\b', user_input, re.IGNORECASE)
            if match_quarter:
                quarter = int(match_quarter.group(1))

        if year is None:
            match_year = re.search(r'\b(20\d{2})\b', user_input)
            if match_year:
                year = int(match_year.group(1))

        return quarter, year

    except Exception as e:
        print(f"Duckling failed: {e}. Falling back to regex extraction.")
        return fallback_extract_entities(user_input)

# Fallback Entity Extraction
def fallback_extract_entities(user_input):
    """
    Fallback mechanism to extract quarter and year using regex.
    """
    quarter = None
    year = None

    # Extract quarter (e.g., "Q1", "Q2", etc.)
    match_quarter = re.search(r'\bQ([1-4])\b', user_input, re.IGNORECASE)
    if match_quarter:
        quarter = int(match_quarter.group(1))

    # Extract year (e.g., "2024")
    match_year = re.search(r'\b(20\d{2})\b', user_input)
    if match_year:
        year = int(match_year.group(1))

    return quarter, year

# Extract Specific Term from User Query
def extract_specific_term(user_input):
    """
    Extract the specific term (e.g., "reported net income") from the user's query.
    """
    terms = ["reported net income", "revenue", "gross profit", "operating income"]
    for term in terms:
        if term in user_input.lower():
            return term
    return "revenue"

# Prediction Function
def predict_value(user_input, quarter, year):
    """
    Predict revenue using the regression model and dynamically adjust the response based on the query.
    """
    if quarter is None or year is None:
        return "Please specify a valid quarter and year."

    # Extract specific term from the query
    specific_term = extract_specific_term(user_input)

    # Generate TF-IDF vector for the input
    input_vector = tfidf_vectorizer.transform([user_input]).toarray()
    features = np.hstack((input_vector, np.array([[quarter, year]])))

    # Predict using the regression model
    prediction = regression_model.predict(features)

    # Reshape and inverse-transform the prediction
    if np.isscalar(prediction):
        prediction = np.array([prediction]).reshape(1, -1)
    else:
        prediction = prediction.reshape(-1, 1)

    original_prediction = scaler_model.inverse_transform(prediction)[0, 0]

    # Return the response dynamically based on the specific term
    return f"The predicted {specific_term} for Q{quarter} {year} is ${original_prediction:,.2f}."

# Fallback Response using DialoGPT
def fallback_response(user_input):
    """
    Generate a natural fallback response using DialoGPT.
    """
    input_ids = dialo_tokenizer.encode(user_input + dialo_tokenizer.eos_token, return_tensors="pt")
    attention_mask = (input_ids != dialo_tokenizer.pad_token_id).long()  # Generate attention mask

    output = dialo_model.generate(
        input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_length=100,
        pad_token_id=dialo_tokenizer.eos_token_id
    )
    response = dialo_tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Chatbot Endpoint
@app.route('/chat', methods=['POST'])
def chatbot():
    """
    Main chatbot logic to handle user queries.
    """
    data = request.json
    user_input = data.get("user_input", "")

    # Recognize intent
    intent = recognize_intent(user_input)

    if intent == "greet":
        response = "Hello! How can I assist you today?"
    elif intent == "exit":
        response = "Goodbye! Feel free to come back anytime."
    elif intent == "predict_revenue":
        quarter, year = extract_entities(user_input)
        if quarter is None or year is None:
            response = "I couldn't extract the quarter or year. Please include both in your query."
        else:
            response = predict_value(user_input, quarter, year)
    else:
        quarter, year = extract_entities(user_input)
        if quarter and year:  # If valid entities are found, predict revenue
            response = predict_value(user_input, quarter, year)
        else:  # Otherwise, fallback to DialoGPT
            response = fallback_response(user_input)

    return jsonify({"response": response})

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=7070)
