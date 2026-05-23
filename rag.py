"""
RAG (Retrieval-Augmented Generation) Module for Clinical Recommendations
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class RAGOutput:
    """Output data structure for RAG module"""
    risk_score: float
    alert: bool
    recommended_actions: List[str]
    explanation: str
    alert_level: str
    monitoring_frequency: str
    patient_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ClinicalRAG:
    """RAG system for clinical recommendations"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.feature_mappings = {
            'HR': 'Heart Rate', 'O2Sat': 'Oxygen Saturation', 'Temp': 'Temperature',
            'SBP': 'Systolic Blood Pressure', 'DBP': 'Diastolic Blood Pressure',
            'Resp': 'Respiratory Rate', 'WBC': 'White Blood Cell Count', 'Lactate': 'Lactate Level'
        }
    
    def _build_knowledge_base(self) -> Dict:
        """Build medical knowledge base"""
        return {
            'HR': {
                'name': 'Heart Rate',
                'actions': ['Assess for arrhythmias', 'Check temperature for fever', 
                           'Evaluate volume status', 'Monitor for signs of shock']
            },
            'SBP': {
                'name': 'Systolic Blood Pressure',
                'actions': ['Start IV fluids - 30 mL/kg crystalloid bolus', 'Obtain blood cultures',
                           'Administer antibiotics within 1 hour', 'Consider vasopressor support']
            },
            'Lactate': {
                'name': 'Lactate Level',
                'actions': ['Repeat lactate measurement in 2-4 hours', 'Aggressive fluid resuscitation',
                           'Ensure adequate tissue perfusion', 'Monitor for organ dysfunction']
            },
            'O2Sat': {
                'name': 'Oxygen Saturation',
                'actions': ['Increase oxygen supplementation', 'Assess respiratory rate and effort',
                           'Obtain arterial blood gas', 'Consider chest X-ray']
            },
            'risk_protocols': {
                'critical': {
                    'threshold': 0.8, 'alert': True, 'alert_level': 'CRITICAL',
                    'monitoring': 'Continuous monitoring every 5 minutes',
                    'actions': ['Activate rapid response team', 'Notify attending physician immediately',
                               'Prepare for ICU transfer', 'Implement sepsis bundle immediately']
                },
                'high': {
                    'threshold': 0.6, 'alert': True, 'alert_level': 'HIGH',
                    'monitoring': 'Monitor vitals every 15 minutes',
                    'actions': ['Notify physician urgently', 'Increase monitoring frequency',
                               'Review recent lab results', 'Prepare for possible escalation']
                },
                'moderate': {
                    'threshold': 0.4, 'alert': False, 'alert_level': 'MODERATE',
                    'monitoring': 'Monitor vitals every 30 minutes',
                    'actions': ['Inform charge nurse', 'Continue close monitoring', 
                               'Document changes in condition']
                },
                'low': {
                    'threshold': 0.0, 'alert': False, 'alert_level': 'LOW',
                    'monitoring': 'Standard monitoring every 1-2 hours',
                    'actions': ['Continue routine care', 'Standard vital sign monitoring']
                }
            }
        }
    
    def get_recommendation(self, risk_score: float, top_features: List[str],
                          patient_id: Optional[str] = None) -> RAGOutput:
        """Generate clinical recommendations"""
        
        if risk_score >= 0.8:
            risk_level = 'critical'
        elif risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        risk_protocol = self.knowledge_base['risk_protocols'][risk_level]
        
        actions = list(risk_protocol['actions'])
        for feature in top_features[:3]:
            if feature in self.knowledge_base:
                actions.extend(self.knowledge_base[feature]['actions'])
        
        seen = set()
        unique_actions = []
        for action in actions:
            if action.lower() not in seen:
                seen.add(action.lower())
                unique_actions.append(action)
        
        feature_names = [self.feature_mappings.get(f, f) for f in top_features[:3]]
        if len(feature_names) >= 2:
            features_text = f"{', '.join(feature_names[:-1])}, and {feature_names[-1]}"
        elif len(feature_names) == 1:
            features_text = feature_names[0]
        else:
            features_text = "multiple vital signs"
        
        if risk_score >= 0.8:
            explanation = f"Critical sepsis risk detected. Abnormal {features_text} indicate severe condition."
        elif risk_score >= 0.6:
            explanation = f"High sepsis risk. Abnormal {features_text} suggest developing sepsis."
        elif risk_score >= 0.4:
            explanation = f"Moderate sepsis risk. {features_text.capitalize()} showing concerning trends."
        else:
            explanation = f"Low sepsis risk. {features_text.capitalize()} within acceptable range."
        
        return RAGOutput(
            risk_score=round(risk_score, 4),
            alert=risk_protocol['alert'],
            recommended_actions=unique_actions[:8],
            explanation=explanation,
            alert_level=risk_protocol['alert_level'],
            monitoring_frequency=risk_protocol['monitoring'],
            patient_id=patient_id
        )
