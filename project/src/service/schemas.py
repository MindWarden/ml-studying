"""Pydantic schemas for the prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
GenderT = Literal["Male", "Female"]
MultipleLinesT = Literal["Yes", "No", "No phone service"]
InternetServiceT = Literal["DSL", "Fiber optic", "No"]
InternetDependentT = Literal["Yes", "No", "No internet service"]
ContractT = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodT = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """Features of a single customer for churn prediction."""

    gender: GenderT
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: MultipleLinesT
    InternetService: InternetServiceT
    OnlineSecurity: InternetDependentT
    OnlineBackup: InternetDependentT
    DeviceProtection: InternetDependentT
    TechSupport: InternetDependentT
    StreamingTV: InternetDependentT
    StreamingMovies: InternetDependentT
    Contract: ContractT
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodT
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_class: YesNo
    threshold: float = 0.5


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
