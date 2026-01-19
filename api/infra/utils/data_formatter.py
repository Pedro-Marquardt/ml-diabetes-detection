from typing import Dict, Any


def format_patient_data(patient_data: Dict[str, Any]) -> str:
    """Formats patient data for LLM prompt"""
    return f"""
PATIENT DATA:
- Age: {patient_data.get('age', 'N/A')} years
- Education level: {patient_data.get('education_level', 'N/A')}
- Income level: {patient_data.get('income_level', 'N/A')}
- Physical activity: {patient_data.get('physical_activity_minutes_per_week', 'N/A')} minutes/week
- Diet score: {patient_data.get('diet_score', 'N/A')}/10
- Family history of diabetes: {'Yes' if patient_data.get('family_history_diabetes', 0) == 1 else 'No'}
- BMI: {patient_data.get('bmi', 'N/A')} kg/m²
- Waist-to-hip ratio: {patient_data.get('waist_to_hip_ratio', 'N/A')}
- Systolic blood pressure: {patient_data.get('systolic_bp', 'N/A')} mmHg
- Total cholesterol: {patient_data.get('cholesterol_total', 'N/A')} mg/dL
- HDL cholesterol: {patient_data.get('hdl_cholesterol', 'N/A')} mg/dL
- LDL cholesterol: {patient_data.get('ldl_cholesterol', 'N/A')} mg/dL
- Triglycerides: {patient_data.get('triglycerides', 'N/A')} mg/dL
- Fasting glucose: {patient_data.get('glucose_fasting', 'N/A')} mg/dL
- Postprandial glucose: {patient_data.get('glucose_postprandial', 'N/A')} mg/dL
- Insulin level: {patient_data.get('insulin_level', 'N/A')} μU/mL
- HbA1c: {patient_data.get('hba1c', 'N/A')}%
- Diabetes risk score: {patient_data.get('diabetes_risk_score', 'N/A')}/10
"""


def format_prediction_result(prediction_result: Dict[str, Any]) -> str:
    """Formats prediction result for LLM prompt"""
    has_diabetes = prediction_result["has_diabetes"]
    probability = prediction_result["probability"]
    confidence = prediction_result["confidence"]
    threshold = prediction_result["threshold_used"]

    probability_percent = probability * 100

    return f"""
ANALYSIS RESULT:
- Prediction: {'DIABETES DETECTED' if has_diabetes else 'Diabetes not detected'}
- Probability: {probability_percent:.2f}%
- Confidence level: {confidence.upper()}
- Threshold used: {threshold:.4f}
"""
