import re
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = "data/tickets.csv"


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    df["text"] = (
        df["subject"].fillna("") + " " +
        df["body"].fillna("")
    )

    df["clean_text"] = df["text"].apply(clean_text)

    return df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=10,
                random_state=42
            )
        )
    ])


# ============================================================
# TRAIN MODEL
# ============================================================

df = load_data()

X = df["clean_text"]
y = df["category"]

model = build_model()

model.fit(X, y)


# ============================================================
# PRIORITY DETECTION
# ============================================================

def detect_priority(text):
    text = text.lower()

    urgent_keywords = [
        "urgent",
        "immediately",
        "down",
        "not working",
        "critical",
        "emergency",
        "asap"
    ]

    for keyword in urgent_keywords:
        if keyword in text:
            return "Urgent"

    return "Normal"


# ============================================================
# TICKET PREDICTION
# ============================================================

def predict_ticket(subject, body):

    raw_text = f"{subject} {body}"
    text = clean_text(raw_text)

    # Empty ticket
    if not text.strip():
        return {
            "category": "General",
            "confidence": 0.0,
            "priority": "Normal",
            "status": "Needs human review"
        }

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = float(max(probabilities))

    priority = detect_priority(text)

    if confidence < 0.60:
        status = "Needs human review"
    else:
        status = "Auto-assigned"

    return {
        "category": prediction,
        "confidence": round(confidence * 100, 2),
        "priority": priority,
        "status": status
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    evaluation_model = build_model()

    evaluation_model.fit(X_train, y_train)

    y_pred = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nTest Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    labels = sorted(y.unique())

    print("\nConfusion Matrix:")
    print("Labels:", labels)
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=labels
        )
    )

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        build_model(),
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    for i, score in enumerate(cv_scores, start=1):
        print(f"Fold {i}: {score:.2%}")

    print(f"\nMean CV Accuracy: {cv_scores.mean():.2%}")
    print(f"CV Standard Deviation: {cv_scores.std():.2%}")


# ============================================================
# COMMAND LINE DEMO
# ============================================================

if __name__ == "__main__":

    evaluate_model()

    sample_tickets = [
        (
            "Payment problem",
            "My card was charged twice for the same purchase."
        ),
        (
            "Login failure",
            "I cannot log into the application because my password is not working."
        ),
        (
            "Leave request",
            "I want to apply for annual leave next week."
        ),
        (
            "Service question",
            "Can you tell me your customer support working hours?"
        ),
        (
            "Critical system issue",
            "The application is down and not working urgently."
        )
    ]

    print("\n" + "=" * 60)
    print("NEW TICKET PREDICTIONS")
    print("=" * 60)

    for subject, body in sample_tickets:

        result = predict_ticket(subject, body)

        print("\nTicket:", subject)
        print("Category:", result["category"])
        print(f"Confidence: {result['confidence']:.2f}%")
        print("Priority:", result["priority"])
        print("Status:", result["status"])