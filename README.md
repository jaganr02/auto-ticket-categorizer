# Auto Email / Ticket Categorizer

An NLP-based support ticket classification and routing system that automatically categorizes incoming support tickets into **Billing, Technical, HR, or General** departments.

The project combines **TF-IDF feature extraction, Logistic Regression, confidence-based routing, priority detection, and a FastAPI-powered web interface** to simulate a lightweight real-time helpdesk triage system.

---

## 1. Project Overview

Support teams often receive a large number of emails and tickets that need to be manually routed to the correct department.

This project automates that initial triage process.

The system:

- Reads the ticket subject and message
- Cleans and preprocesses the text
- Converts text into numerical TF-IDF features
- Predicts the appropriate department using Logistic Regression
- Calculates prediction confidence
- Detects ticket priority
- Sends low-confidence predictions for human review
- Provides a live web interface for real-time classification

### Supported Categories

- **Billing**
- **Technical**
- **HR**
- **General**

---

## 2. Key Features

### Core ML Features

- Text preprocessing
- Subject + body text combination
- TF-IDF vectorization
- Unigram and bigram features
- Logistic Regression classification
- Stratified train/test split
- 5-fold cross-validation
- Accuracy evaluation
- Precision, Recall and F1-score
- Confusion matrix

### Intelligent Routing Features

- Confidence score
- 60% human-review threshold
- Urgent / Normal priority detection
- Empty-ticket handling
- Automatic department routing

### Live Demo

A professional web interface is included using:

- HTML
- CSS
- JavaScript
- FastAPI

The frontend sends the ticket to the backend and displays the prediction instantly.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Ticket classification |
| FastAPI | Backend API |
| Uvicorn | API server |
| HTML | Frontend structure |
| CSS | Frontend styling |
| JavaScript | Frontend interaction |
| Matplotlib | Visualization |
| Seaborn | Visualization |

---

## 4. Project Architecture

```text
Incoming Ticket
      │
      ▼
Subject + Body
      │
      ▼
Text Preprocessing
      │
      ├── Lowercase
      ├── Remove special characters
      └── Remove extra whitespace
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
      │
      ├── Category
      ├── Confidence
      ├── Priority
      └── Routing Status
             │
             ├── ≥ 60% → Auto-assigned
             │
             └── < 60% → Human Review
