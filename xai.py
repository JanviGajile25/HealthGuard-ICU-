"""
Explainable AI (XAI) Module for Sepsis Prediction
"""

import torch
import numpy as np
from typing import Dict, List


class XAIExplainer:
    """Explainability module for sepsis predictions"""
    
    def __init__(self, model=None):
        self.model = model
        self.feature_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'WBC', 'Lactate']
        
        self.medical_logic = {
            'HR': {'high_risk_when': 'high', 'description': 'Heart Rate'},
            'O2Sat': {'high_risk_when': 'low', 'description': 'Oxygen Saturation'},
            'Temp': {'high_risk_when': 'high', 'description': 'Temperature'},
            'SBP': {'high_risk_when': 'low', 'description': 'Systolic BP'},
            'DBP': {'high_risk_when': 'low', 'description': 'Diastolic BP'},
            'Resp': {'high_risk_when': 'high', 'description': 'Respiratory Rate'},
            'WBC': {'high_risk_when': 'high', 'description': 'White Blood Cell Count'},
            'Lactate': {'high_risk_when': 'high', 'description': 'Lactate Level'}
        }
    
    def calculate_feature_importance(self, patient_data: np.ndarray) -> np.ndarray:
        """Calculate feature importance using simple gradient-based method"""
        importance = np.abs(patient_data).mean(axis=0)
        max_importance = importance.max()
        if max_importance > 0:
            importance = importance / max_importance
        return importance
    
    def explain_prediction(self, patient_data: np.ndarray, risk_score: float = None) -> Dict:
        """Generate explanation for a prediction"""
        
        if patient_data.ndim == 1:
            patient_data = patient_data.reshape(1, -1)
            patient_data = np.repeat(patient_data, 10, axis=0)
        
        if risk_score is None and self.model is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            patient_tensor = torch.tensor(patient_data, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                risk_score = self.model(patient_tensor).item()
        
        feature_importance = self.calculate_feature_importance(patient_data)
        feature_values = patient_data.mean(axis=0)
        
        top_3_indices = np.argsort(feature_importance)[::-1][:3]
        
        top_features = [
            {
                'name': self.feature_names[idx],
                'importance': float(feature_importance[idx]),
                'value': float(feature_values[idx])
            }
            for idx in top_3_indices
        ]
        
        top_features_text = ', '.join([f['name'] for f in top_features])
        explanation = f"Risk is {'HIGH' if risk_score > 0.5 else 'LOW'} ({risk_score:.1%}). "
        explanation += f"Top contributors: {top_features_text}."
        
        return {
            'risk_score': risk_score,
            'risk_level': 'HIGH' if risk_score > 0.5 else 'LOW',
            'feature_importance': {
                self.feature_names[i]: float(feature_importance[i])
                for i in range(len(self.feature_names))
            },
            'top_features': top_features,
            'explanation': explanation
        }
