from evaluate import load
import requests
import matplotlib.pyplot as plt

bleu = load("bleu")

test_cases = [
    {"query": "What is the revenue for Q3 2024?", "expected_response": "The predicted revenue for Q3 2024 is $4,539.97."},
    {"query": "Reported net income Q3 2024", "expected_response": "The predicted reported net income for Q3 2024 is $1,733.10."},
    {"query": "Hello", "expected_response": "Hello! How can I assist you today?"},
    {"query": "Goodbye", "expected_response": "Goodbye! Feel free to come back anytime."},
    {"query": "Revenue next quarter", "expected_response": "I couldn't extract the quarter or year. Please include both in your query."},
]

def get_chatbot_response(query):
    url = "http://127.0.0.1:8080/chat"
    try:
        response = requests.post(url, json={"user_input": query})
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error: Received status code {response.status_code}")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return ""

predictions = []
references = []

for test in test_cases:
    chatbot_response = get_chatbot_response(test["query"])
    print(f"Query: {test['query']}")
    print(f"Chatbot Response: {chatbot_response}")
    print(f"Expected Response: {test['expected_response']}")
    
    predictions.append(chatbot_response)
    references.append([test["expected_response"]])

try:
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"\nBLEU Score: {bleu_score['bleu']:.4f}")
except Exception as e:
    print(f"Error computing BLEU score: {e}")