# CIBC Quarterly Report Chatbot

Welcome to the **CIBC Quarterly Report Chatbot** repository! This project leverages Natural Language Processing (NLP) to provide users with a conversational interface for accessing insights from CIBC's quarterly financial reports. The chatbot enables users to query specific data, trends, and analyses from these reports quickly and efficiently.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
7. [Team Members](#team-members)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

---

## Overview

The **CIBC Quarterly Report Chatbot** simplifies financial report analysis by enabling users to:
- Ask questions about CIBC’s financial data.
- Retrieve insights on key performance indicators (KPIs) such as revenue, net income, and expenses.
- Compare trends across different quarters or years.

This project was developed as part of a larger initiative to streamline financial data accessibility for non-technical stakeholders.

---

## Features

- **Conversational Interface**: Users can interact with the chatbot in plain English.
- **NLP Capabilities**: Employs advanced NLP techniques to extract and summarize information from quarterly reports.
- **Custom Query Handling**: Processes user queries to provide precise and contextually relevant answers.
- **Data Visualization**: Displays trends, graphs, and other visual insights where applicable.

---

## Technologies Used

- **Programming Language**: Python
- **Natural Language Processing**: Hugging Face Transformers, spaCy
- **Frameworks**: Flask (Backend), React (Frontend)
- **Data Storage**: SQLite, Pandas
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Docker, AWS

---

## Setup and Installation

### Prerequisites
- Python 3.8 or later
- Node.js and npm
- Docker (optional, for containerized deployment)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/aditi-pithva/bank-quarterly-report-chatbot.git
   cd bank-quarterly-report-chatbot
   ```
2. Run news-release.ipynb
3. Copy all the mdoels generated to cibc-finacial-insight-backend/models
4. Install backend dependencies:
   ```bash
   cd cibc-finacial-insight-backend
   pip install -r requirements.txt
   ```
5. Install frontend dependencies:
   ```bash
   cd cibc-finacial-insight-backend
   npm install
   ```
6. Start the backend server:
   ```bash
   python app.py
   ```
7. Access the chatbot at `http://localhost:8000`.

---

## Usage

1. Launch the chatbot in your browser.
2. Upload CIBC quarterly reports in `.pdf` or `.docx` format.
3. Interact with the chatbot by asking questions such as:
   - "What was the revenue for Q3 2023?"
   - "Show the trend in net income for the last four quarters."
   - "Compare Q1 and Q2 expenses for 2024."

The chatbot will parse the report, extract relevant data, and provide answers in text or graphical form.

---

## Model Training and Fine-Tuning

The chatbot’s NLP capabilities are powered by a fine-tuned transformer model. Here are the steps followed for training:

1. **Data Preprocessing**: Extracted text and structured data from quarterly reports.
2. **Tokenization**: Processed using Hugging Face’s tokenizers.
3. **Fine-Tuning**: Fine-tuned a pre-trained BERT-based model on a dataset of financial queries and answers.
4. **Evaluation**: Tested using real-world queries to ensure accuracy and relevance.

---

## Acknowledgments

This project was completed with the guidance and support of **Ashish Gupta**, who provided valuable insights and feedback throughout the development process.

Special thanks to CIBC for providing access to the quarterly reports used in testing and validation.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For further inquiries or contributions, please feel free to reach out or create an issue in this repository.

