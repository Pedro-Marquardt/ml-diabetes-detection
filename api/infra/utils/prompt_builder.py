from typing import Dict, Any
from api.infra.utils.data_formatter import format_patient_data, format_prediction_result


def create_system_prompt() -> str:
    """Creates system prompt for LLM"""
    return """You are an endocrinologist and diabetes specialist. Analyze patient clinical data and ML prediction results to generate a clear, professional diagnostic report. Explain the prediction meaning, highlight key risk factors, provide practical recommendations, and emphasize regular medical follow-up. Keep it evidence-based and empathetic."""


def create_user_prompt(
    patient_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
) -> str:
    """Creates user prompt with patient data and prediction result"""
    patient_info = format_patient_data(patient_data)
    prediction_info = format_prediction_result(prediction_result)
    
    return f"""{patient_info}

{prediction_info}

Please generate a comprehensive and explanatory medical report for this patient, explaining the analysis result, the main risk factors identified, and practical recommendations for diabetes prevention or control."""
