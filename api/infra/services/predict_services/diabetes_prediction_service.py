import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


class DiabetesPredictionService:
    
    def __init__(self):

        service_dir = Path(__file__).resolve().parent  
        infra_dir = service_dir.parent.parent  
        self.model_path = infra_dir / "models" / "model_optimized" / "diabetes_model_optimized.joblib"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model_package = self._load_model()
        self.model = self.model_package['model']
        self.threshold = self.model_package.get('threshold', 0.5)
        self.preprocessors = self.model_package.get('preprocessors', {})
        
    def _load_model(self) -> Dict[str, Any]:
        return joblib.load(self.model_path)
    
    def _preprocess(self, patient_data: Dict[str, Any]) -> np.ndarray:

        feature_order = [
            'age', 'education_level', 'income_level',
            'physical_activity_minutes_per_week', 'diet_score',
            'family_history_diabetes', 'bmi', 'waist_to_hip_ratio',
            'systolic_bp', 'cholesterol_total', 'hdl_cholesterol',
            'ldl_cholesterol', 'triglycerides', 'glucose_fasting',
            'glucose_postprandial', 'insulin_level', 'hba1c',
            'diabetes_risk_score'
        ]
        
        df = pd.DataFrame([patient_data])[feature_order]
        
        if 'label_encoders' in self.preprocessors:
            label_encoders = self.preprocessors['label_encoders']
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col].astype(str))
        
        if 'scaler' in self.preprocessors:
            scaler = self.preprocessors['scaler']
            scaled = scaler.transform(df)
        else:
            scaled = df.values
            
        return scaled
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        X_processed = self._preprocess(patient_data)
        
        # Garantir que X_processed está no formato correto (2D array)
        if X_processed.ndim == 1:
            X_processed = X_processed.reshape(1, -1)
        
        # Obter probabilidades
        try:
            probabilities = self.model.predict_proba(X_processed)
            probability = probabilities[0, 1] if probabilities.shape[1] > 1 else probabilities[0, 0]
        except AttributeError as e:
            # Fallback se predict_proba não estiver disponível
            prediction = self.model.predict(X_processed)[0]
            probability = float(prediction)
        
        has_diabetes = probability >= self.threshold
        
        distance_from_threshold = abs(probability - self.threshold)
        if distance_from_threshold > 0.2:
            confidence = "high"
        elif distance_from_threshold > 0.1:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "has_diabetes": bool(has_diabetes),
            "probability": float(probability),
            "threshold_used": float(self.threshold),
            "confidence": confidence
        }
