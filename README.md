

# Quarterly Report Chatbot

Welcome to the **Quarterly Report Chatbot** repository! This project uses advanced Natural Language Processing (NLP) techniques to help users interact with **CIBC’s quarterly financial reports** via a conversational interface. Users can query specific KPIs or request **predictions for future quarters**.

---

![image](https://github.com/user-attachments/assets/8c275e16-6162-4072-af8a-5e2720948e14)


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Team Members](#team-members)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This chatbot simplifies financial report analysis by allowing users to:

- Ask questions about historical financial data
- Predict future KPIs such as revenue and net income (e.g., Q4 results)
- Retrieve insights across quarters using natural language

---

## Features

- **Conversational Interface**: Seamless interaction in plain English
- **Regression-Based Prediction**: Predicts Q4 values using pre-trained regression models
- **Intent Recognition**: Classifies queries using BERT-based text classification
- **Quarter & Year Detection**: Extracts temporal context using Duckling and regex
- **BLEU-Based Evaluation**: Validates response accuracy through BLEU score
- **Fallback Response Handling**: Uses DialoGPT to respond when no structured intent is found

---

## 🛠 Technologies Used

- **Language & Frameworks**: Python, Flask
- **NLP Models**: BERT (for intent classification), DialoGPT (for fallback), TF-IDF
- **Machine Learning**: XGBoost (for KPI prediction), scikit-learn, joblib
- **Utilities**: Duckling for entity extraction, Hugging Face Transformers
- **Visualization & Evaluation**: Matplotlib, BLEU score (`evaluate` library)
- **Storage**: SQLite, Pandas
- **Frontend**: HTML (via Flask `render_template`)
- **Deployment (Local)**: Flask development server (no cloud deployment yet)

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js and npm (optional for frontend)
- Docker (optional)
  
### Installation Steps
```bash
# Clone the repository
git clone https://github.com/aditi-pithva/bank-quarterly-report-chatbot.git
cd bank-quarterly-report-chatbot

# Run model training and save outputs
# (Inside Jupyter or equivalent)
Open `model-training.ipynb` and execute all cells

# Copy models to backend
cp -r models/ cibc-finacial-insight-backend/models

# Install backend dependencies
cd cibc-finacial-insight-backend
pip install -r requirements.txt

# Run the Flask backend server
python app.py
```

Access the chatbot at: `http://localhost:8080`

---

## Usage

Ask queries like:

- “What was the revenue for Q3 2023?”
- “Predict net income for Q4 2024.”
- “Compare revenue between Q1 and Q2 of 2023.”

The chatbot will:

- Detect your intent (predict/query/fallback)
- Extract quarter/year using Duckling or regex
- Run prediction or retrieve answers
- Reply in natural language

---

## Model Training

Training steps used in `model-training.ipynb`:

1. **Data Preprocessing**: Parsed and cleaned financial reports from PDFs
2. **TF-IDF Vectorization**: Applied to convert text queries into numerical features
3. **Model Training**: Trained XGBoost regression model on historical KPI values
4. **Intent Recognition**: Used Hugging Face’s BERT model for binary intent classification (`predict_revenue` vs. other)
5. **Scaler**: MinMaxScaler used for normalizing regression output

---

## Evaluation

- BLEU score evaluation is implemented in `bleu.py`
- Tests chatbot's actual vs. expected response for multiple queries
- Example:
```bash
python bleu.py
```

---

## Team Members
- Aagam Shah 
- Aditi Pithva
- Daivik Pelathur

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
