"""
ICU Sepsis Risk Prediction System - Main Entry Point
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import SepsisDataProcessor
from src.model import train_model, load_model, predict
from src.alert import SmartAlertSystem
from src.xai import XAIExplainer
from src.rag import ClinicalRAG


def preprocess_data():
    """Run preprocessing pipeline"""
    print("="*70)
    print("PREPROCESSING DATA")
    print("="*70)
    
    # Check if preprocessed data already exists
    if (os.path.exists('data/X_train.npy') and 
        os.path.exists('data/y_train.npy') and
        os.path.exists('data/X_test.npy') and
        os.path.exists('data/y_test.npy')):
        print("✓ Preprocessed data already exists in data/ directory")
        response = input("Do you want to reprocess? (y/n): ").lower()
        if response != 'y':
            print("Skipping preprocessing. Using existing data.")
            X_train = np.load('data/X_train.npy')
            y_train = np.load('data/y_train.npy')
            X_test = np.load('data/X_test.npy')
            y_test = np.load('data/y_test.npy')
            print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            return X_train, y_train, X_test, y_test
    
    processor = SepsisDataProcessor(window_size=10)
    X_train, y_train, X_test, y_test = processor.run(
        data_dir="./data/training",
        fallback_csv="d:/ICU/icu_sepsis_dataset.csv/Dataset.csv"
    )
    
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("\n✓ Preprocessing complete\n")
    
    return X_train, y_train, X_test, y_test


def train():
    """Train the model"""
    print("="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    if not os.path.exists('data/X_train.npy'):
        print("Preprocessed data not found. Running preprocessing...")
        preprocess_data()
    
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    model = train_model(X_train, y_train, X_test, y_test, epochs=10)
    
    print("\n✓ Training complete\n")
    return model


def predict_patient(patient_data=None):
    """Make prediction for a patient"""
    print("="*70)
    print("PATIENT RISK PREDICTION")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('data/sepsis_lstm.pth'):
        print("❌ Model not found. Training model first...")
        train()
    
    # Load model
    model = load_model('data/sepsis_lstm.pth')
    
    # Use test data if no patient data provided
    if patient_data is None:
        if not os.path.exists('data/X_test.npy'):
            print("❌ Test data not found. Running preprocessing first...")
            preprocess_data()
        X_test = np.load('data/X_test.npy')
        patient_data = X_test[0]
    
    # Make prediction
    risk_score = predict(model, patient_data)
    
    # Get explanation
    explainer = XAIExplainer(model)
    explanation = explainer.explain_prediction(patient_data, risk_score)
    
    # Get recommendations
    rag = ClinicalRAG()
    top_features = [f['name'] for f in explanation['top_features']]
    recommendation = rag.get_recommendation(risk_score, top_features, patient_id="ICU-001")
    
    # Display results
    print(f"\n📊 Risk Score: {risk_score:.1%}")
    print(f"📊 Risk Level: {explanation['risk_level']}")
    print(f"\n🔍 Top Contributing Features:")
    for i, feature in enumerate(explanation['top_features'], 1):
        print(f"  {i}. {feature['name']}: {feature['importance']:.1%} importance")
    
    print(f"\n💬 Explanation: {explanation['explanation']}")
    
    print(f"\n💊 Recommended Actions:")
    for i, action in enumerate(recommendation.recommended_actions[:5], 1):
        print(f"  {i}. {action}")
    
    print(f"\n⚠️  Alert Level: {recommendation.alert_level}")
    print(f"📅 Monitoring: {recommendation.monitoring_frequency}")
    
    print("\n✓ Prediction complete\n")
    
    return risk_score, explanation, recommendation


def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("="*70)
    print("LAUNCHING DASHBOARD")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('data/sepsis_lstm.pth'):
        print("⚠️  Warning: Model not found. Dashboard will work but predictions may be random.")
        print("   Run 'python main.py train' first for accurate predictions.\n")
    
    # Check if data exists
    if not os.path.exists('data/X_test.npy'):
        print("⚠️  Warning: Test data not found. Dashboard will use simulated data only.")
        print("   Run 'python main.py preprocess' first for real patient data.\n")
    
    print("\nStarting Streamlit dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")
    
    os.system("python -m streamlit run src/dashboard.py")


def main():
    """Main entry point"""
    print("\n")
    print("="*70)
    print(" "*15 + "ICU SEPSIS RISK PREDICTION SYSTEM")
    print("="*70)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py preprocess    - Preprocess data")
        print("  python main.py train         - Train model")
        print("  python main.py predict       - Make prediction")
        print("  python main.py dashboard     - Launch dashboard")
        print("  python main.py all           - Run complete pipeline")
        print()
        print("Quick Start:")
        print("  1. First time: python main.py all")
        print("  2. After that: python main.py dashboard (or predict)")
        print()
        print("Status:")
        data_exists = os.path.exists('data/X_train.npy')
        model_exists = os.path.exists('data/sepsis_lstm.pth')
        print(f"  Data preprocessed: {'✓ Yes' if data_exists else '✗ No (run preprocess)'}")
        print(f"  Model trained:     {'✓ Yes' if model_exists else '✗ No (run train)'}")
        print()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'preprocess':
        preprocess_data()
    
    elif command == 'train':
        train()
    
    elif command == 'predict':
        predict_patient()
    
    elif command == 'dashboard':
        run_dashboard()
    
    elif command == 'all':
        print("Running complete pipeline...\n")
        preprocess_data()
        train()
        predict_patient()
        print("\nTo launch dashboard, run: python main.py dashboard")
    
    else:
        print(f"Unknown command: {command}")
        print("Use: preprocess, train, predict, dashboard, or all")


if __name__ == "__main__":
    main()
