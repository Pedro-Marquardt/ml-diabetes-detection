from pydantic import BaseModel, Field
from typing import Literal
from api.application.enum.education_level import EducationLevel
from api.application.enum.income_level import IncomeLevel


class PatientData(BaseModel):
    """Dados do paciente para predição - 18 features"""

    age: float = Field(..., ge=0, le=120, description="Idade do paciente")
    education_level: EducationLevel = Field(..., description="Nível de educação")
    income_level: IncomeLevel = Field(..., description="Nível de renda")
    physical_activity_minutes_per_week: float = Field(..., ge=0)
    diet_score: float = Field(..., ge=0, le=10)
    family_history_diabetes: int = Field(..., ge=0, le=1)
    bmi: float = Field(..., ge=0)
    waist_to_hip_ratio: float = Field(..., ge=0)
    systolic_bp: float = Field(..., ge=0)
    cholesterol_total: float = Field(..., ge=0)
    hdl_cholesterol: float = Field(..., ge=0)
    ldl_cholesterol: float = Field(..., ge=0)
    triglycerides: float = Field(..., ge=0)
    glucose_fasting: float = Field(..., ge=0)
    glucose_postprandial: float = Field(..., ge=0)
    insulin_level: float = Field(..., ge=0)
    hba1c: float = Field(..., ge=0)
    diabetes_risk_score: float = Field(..., ge=0, le=10)


class PredictionResponse(BaseModel):
    """Resposta da predição"""

    has_diabetes: bool
    probability: float = Field(..., ge=0, le=1)
    threshold_used: float
    confidence: Literal["high", "medium", "low"]


class DiagnosticReportResponse(BaseModel):
    """Resposta completa com predição e relatório diagnóstico"""

    prediction: PredictionResponse
    diagnostic_report: str = Field(
        ..., description="Relatório médico explicativo gerado pela LLM"
    )
