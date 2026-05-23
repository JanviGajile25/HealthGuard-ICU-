# ICU Sepsis Risk Prediction System

AI-powered system for real-time sepsis risk prediction in ICU patients using LSTM neural networks with explainable AI and clinical recommendations.

## Features

- **LSTM Model**: Deep learning model for time-series sepsis risk prediction
- **Real-Time Monitoring**: Live dashboard with continuous patient monitoring
- **Explainable AI (XAI)**: Feature importance and medical logic validation
- **Clinical Recommendations**: RAG-based actionable clinical guidelines
- **Smart Alerts**: Multi-condition alert system (high risk + persistence + trend)

## Quick Start

### First Time Setup (One Time Only)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline (preprocess + train)
python main.py all
```

This will:
- Preprocess the data (creates data/X_train.npy, etc.)
- Train the model (creates data/sepsis_lstm.pth)
- Make a sample prediction

**You only need to do this ONCE!**

### Daily Usage (After Setup)

```bash
# Launch dashboard (most common)
python main.py dashboard

# Or launch dashboard directly with Streamlit
python -m streamlit run src/dashboard.py

# Or make predictions
python main.py predict
```

**No need to preprocess or train again!** The system will use the existing data and model.

### When to Re-run Setup

Only re-run preprocessing/training if:
- You have new patient data
- You want to retrain the model
- Data files are missing or corrupted

### Check Status

```bash
# Run without arguments to see status
python main.py
```

This shows:
- ✓ Data preprocessed: Yes/No
- ✓ Model trained: Yes/No

## Project Structure

```
project/
├── data/                      # Data directory
│   ├── X_train.npy           # Preprocessed training data
│   ├── y_train.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   └── sepsis_lstm.pth       # Trained model
│
├── src/                       # Source code
│   ├── preprocess.py         # Data preprocessing
│   ├── model.py              # LSTM model
│   ├── alert.py              # Smart alert system
│   ├── xai.py                # Explainability module
│   ├── rag.py                # Clinical recommendations
│   └── dashboard.py          # Streamlit dashboard
│
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Features Explained

### 8 Clinical Features

- **HR**: Heart Rate (bpm)
- **O2Sat**: Oxygen Saturation (%)
- **Temp**: Body Temperature (°C)
- **SBP**: Systolic Blood Pressure (mmHg)
- **DBP**: Diastolic Blood Pressure (mmHg)
- **Resp**: Respiratory Rate (breaths/min)
- **WBC**: White Blood Cell Count (K/μL)
- **Lactate**: Lactate Level (mmol/L)

### Risk Levels

- **Low** (0-40%): Standard monitoring
- **Moderate** (40-60%): Increased monitoring
- **High** (60-80%): Urgent attention required
- **Critical** (80-100%): Immediate intervention

### Alert Conditions

Alert triggers when ALL conditions are met:
1. **High Risk**: Current risk > 80%
2. **Persistence**: Risk > 80% for 10+ consecutive steps
3. **Increasing Trend**: Risk increasing over last 3 steps

## Dashboard Features

- **Real-Time Monitoring**: Live vital signs with auto-update
- **Simulation Modes**: Normal variation, deterioration, manual control
- **Visual Analytics**: Multi-panel charts for all vitals
- **Risk Tracking**: Continuous risk score monitoring
- **Alert History**: Log of all critical alerts
- **Clinical Actions**: Recommended interventions based on risk level

## Model Architecture

```
Input (8 features, 10 timesteps)
    ↓
LSTM Layer (64 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit, Sigmoid)
    ↓
Risk Score (0-1)
```

## Usage Examples

### Python API

```python
from src.model import load_model, predict
from src.xai import XAIExplainer
from src.rag import ClinicalRAG

# Load model
model = load_model('data/sepsis_lstm.pth')

# Make prediction
risk_score = predict(model, patient_data)

# Get explanation
explainer = XAIExplainer(model)
explanation = explainer.explain_prediction(patient_data, risk_score)

# Get recommendations
rag = ClinicalRAG()
recommendation = rag.get_recommendation(
    risk_score=risk_score,
    top_features=['HR', 'SBP', 'Lactate'],
    patient_id='ICU-001'
)
```

### Command Line

```bash
# Preprocess data
python main.py preprocess

# Train model
python main.py train

# Make prediction
python main.py predict

# Launch dashboard (option 1)
python main.py dashboard

# Launch dashboard (option 2 - direct)
python -m streamlit run src/dashboard.py

# Run complete pipeline
python main.py all
```

## Data Format

Input data should be in PSV (pipe-separated values) format with columns:
- Patient_ID
- HR, O2Sat, Temp, SBP, DBP, Resp, WBC, Lactate
- SepsisLabel (0 or 1)

## Performance

- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~98%
- **Inference Time**: <10ms per patient
- **Dashboard Update**: Real-time (0.5-5s intervals)

## Clinical Validation

- Feature importance aligns with medical knowledge
- Recommendations follow sepsis bundle guidelines
- Alert system reduces false positives through multi-condition logic

## License

This is a research/educational project. Not for clinical use without proper validation and regulatory approval.

## Support

For issues or questions, please refer to the code documentation or create an issue in the repository.
