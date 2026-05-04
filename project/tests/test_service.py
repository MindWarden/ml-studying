"""End-to-end checks for the FastAPI service."""
from fastapi.testclient import TestClient

from src.service.app import app

LOYAL_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 100.0,
    "TotalCharges": 6000.0,
}

RISKY_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 75.0,
}


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_returns_valid_response():
    with TestClient(app) as client:
        r = client.post("/predict", json=LOYAL_CUSTOMER)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_class"] in ("Yes", "No")


def test_predict_risky_higher_than_loyal():
    with TestClient(app) as client:
        loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()
        risky = client.post("/predict", json=RISKY_CUSTOMER).json()
        assert risky["churn_probability"] > loyal["churn_probability"]


def test_predict_validation_error_on_bad_input():
    with TestClient(app) as client:
        bad = {**LOYAL_CUSTOMER, "Contract": "InvalidValue"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422
