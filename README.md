# Auto Email / Ticket Categorizer

## 1. Project Overview

This project is an NLP-based support ticket classification system that automatically categorizes incoming tickets into four departments:

* Billing
* Technical
* HR
* General

The system combines text preprocessing, TF-IDF feature extraction, and Logistic Regression to classify support tickets.

It also provides confidence scoring, priority detection, and a human-review fallback for low-confidence predictions.

## 2. Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Matplotlib / Seaborn for visualization if required

## 3. Project Structure

```text
ticket-categorizer/
│
├── data/
│   └── tickets.csv
│
├── ticket_categorizer.py
├── requirements.txt
└── README.md
```

## 4. Dataset

The development dataset contains 80 labeled support tickets distributed equally across four categories:

* Billing: 20
* Technical: 20
* HR: 20
* General: 20

Each ticket contains:

* Subject
* Body
* Category

The subject and body are combined before text preprocessing.

## 5. Text Preprocessing

The preprocessing pipeline performs the following steps:

1. Convert text to lowercase.
2. Remove special characters.
3. Remove extra whitespace.
4. Combine the email subject and body.
5. Convert the cleaned text into TF-IDF features.

## 6. Model

The project uses a Scikit-learn Pipeline containing:

```text
TF-IDF Vectorizer
        ↓
Logistic Regression
```

Logistic Regression was selected because it is simple, fast, interpretable, and effective for text classification problems with a relatively small dataset.

The TF-IDF vectorizer uses unigram and bigram features to capture both individual words and short phrases.

## 7. Model Evaluation

The dataset is divided into training and testing sets using a stratified split.

The current development results are:

* Test Accuracy: 85%
* Mean 5-Fold Cross-Validation Accuracy: 90%

The model is also evaluated using:

* Precision
* Recall
* F1-score
* Confusion Matrix

## 8. Confidence-Based Routing

The classifier uses prediction probabilities to calculate a confidence score.

If the highest prediction probability is below 60%, the ticket is not automatically assigned.

Instead, the system returns:

```text
Needs human review
```

This helps prevent unreliable automatic routing.

## 9. Priority Detection

A simple rule-based priority detector identifies urgent tickets using keywords such as:

* urgent
* immediately
* down
* not working
* critical
* emergency
* asap

Tickets containing these keywords are marked as:

```text
Urgent
```

Other tickets are marked:

```text
Normal
```

## 10. Example Predictions

Example results from the development dataset:

| Ticket                | Category  | Confidence | Priority |
| --------------------- | --------- | ---------: | -------- |
| Payment problem       | Billing   |     90.69% | Normal   |
| Login failure         | Technical |     82.19% | Urgent   |
| Leave request         | HR        |     83.75% | Normal   |
| Service question      | General   |     85.94% | Normal   |
| Critical system issue | Technical |     66.26% | Urgent   |

## 11. Edge Cases

The system handles several edge cases:

* Empty ticket text
* Low-confidence predictions
* Urgent tickets
* Unexpected or ambiguous ticket content

For an empty ticket, the system returns General with zero confidence and sends the ticket for human review.

For predictions below the 60% confidence threshold, the system also sends the ticket for human review.

## 12. How to Run

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classifier:

```bash
python ticket_categorizer.py
```

## 13. Future Improvements

Possible improvements include:

* Using a larger real-world labeled dataset
* Adding more advanced NLP preprocessing
* Testing additional classification algorithms
* Using transformer-based text embeddings
* Adding a Streamlit interface
* Storing prediction history
* Adding feedback-based model retraining
* Improving urgency detection using a trained classifier

## 14. Conclusion

This project demonstrates how NLP and machine learning can automate support-ticket routing. TF-IDF combined with Logistic Regression provides a lightweight and effective solution, while confidence-based human review helps reduce incorrect automatic assignments.
