"""
Smart Alert System for ICU Risk Prediction
"""

from typing import Dict


class SmartAlertSystem:
    """Smart alert system for ICU patient risk prediction"""
    
    def __init__(self, risk_threshold=0.8, persistence_steps=10, increasing_steps=3):
        self.risk_threshold = risk_threshold
        self.persistence_steps = persistence_steps
        self.increasing_steps = increasing_steps
        self.risk_history = []
        
    def reset(self):
        """Reset the alert system history"""
        self.risk_history = []
    
    def get_risk_level(self, risk: float) -> str:
        """Categorize risk level"""
        if risk <= 0.4:
            return "Low"
        elif risk <= 0.7:
            return "Medium"
        else:
            return "High"
    
    def check_persistence(self) -> bool:
        """Check if risk has persisted above threshold"""
        if len(self.risk_history) < self.persistence_steps:
            return False
        recent = self.risk_history[-self.persistence_steps:]
        return all(score > self.risk_threshold for score in recent)
    
    def check_increasing_trend(self) -> bool:
        """Check if risk is strictly increasing"""
        if len(self.risk_history) < self.increasing_steps:
            return False
        trend_window = self.risk_history[-self.increasing_steps:]
        return all(trend_window[i] < trend_window[i+1] 
                  for i in range(len(trend_window) - 1))
    
    def process_risk_score(self, current_risk: float) -> Dict:
        """Process a new risk score and determine alert status"""
        self.risk_history.append(current_risk)
        
        risk_level = self.get_risk_level(current_risk)
        is_high_risk = current_risk > self.risk_threshold
        is_persistent = self.check_persistence()
        is_increasing = self.check_increasing_trend()
        
        alert_triggered = is_high_risk and is_persistent and is_increasing
        
        if alert_triggered:
            message = "🚨 CRITICAL ALERT: High sepsis risk detected!"
        elif is_high_risk and is_persistent:
            message = "⚠️  WARNING: High risk persisting, monitoring trend..."
        elif is_high_risk:
            message = "⚠️  CAUTION: High risk detected, monitoring persistence..."
        elif risk_level == "Medium":
            message = "ℹ️  NOTICE: Medium risk, continue monitoring"
        else:
            message = "✅ STABLE: Patient condition stable"
        
        return {
            "alert_triggered": alert_triggered,
            "current_risk": current_risk,
            "risk_level": risk_level,
            "is_high_risk": is_high_risk,
            "is_persistent": is_persistent,
            "is_increasing": is_increasing,
            "message": message,
            "history_length": len(self.risk_history)
        }
