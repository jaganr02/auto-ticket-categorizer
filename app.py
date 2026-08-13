from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ticket_categorizer import predict_ticket


app = FastAPI(
    title="Auto Ticket Categorizer API",
    description="NLP-based support ticket classification and routing system",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TicketRequest(BaseModel):
    subject: str
    body: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Auto Ticket Categorizer"
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict(request: TicketRequest):

    if not request.subject.strip() and not request.body.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a subject or ticket message."
        )

    result = predict_ticket(
        request.subject,
        request.body
    )

    return result


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)