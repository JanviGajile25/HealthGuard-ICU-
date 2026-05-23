"""
Real-Time ICU Monitoring Dashboard
Simulates live patient monitoring with dynamic updates
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Try to import RAG module
try:
    from src.rag import ClinicalRAG
    HAS_RAG = True
except ImportError:
    try:
        from rag import ClinicalRAG
        HAS_RAG = True
    except ImportError:
        HAS_RAG = False

# Page configuration
st.set_page_config(
    page_title="Real-Time ICU Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    .alert-high {
        background-color: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .vital-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'data_history' not in st.session_state:
    st.session_state.data_history = {
        'time': [],
        'HR': [],
        'SBP': [],
        'DBP': [],
        'SpO2': [],
        'Resp': [],
        'Temp': []
    }
if 'current_vitals' not in st.session_state:
    st.session_state.current_vitals = {
        'HR': 75,
        'SBP': 120,
        'DBP': 80,
        'SpO2': 98,
        'Resp': 16,
        'Temp': 37.0
    }
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0.25
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'time_elapsed' not in st.session_state:
    st.session_state.time_elapsed = 0

# Simulation parameters
NORMAL_RANGES = {
    'HR': (60, 100),
    'SBP': (90, 140),
    'DBP': (60, 90),
    'SpO2': (95, 100),
    'Resp': (12, 20),
    'Temp': (36.5, 37.5)
}

def simulate_vital_change(current_value, vital_name, deterioration_mode=False):
    """Simulate realistic vital sign changes"""
    min_val, max_val = NORMAL_RANGES[vital_name]
    
    if deterioration_mode:
        # Simulate deterioration
        if vital_name in ['HR', 'Resp', 'Temp']:
            # These increase during sepsis
            change = np.random.uniform(0.5, 2.0)
        elif vital_name in ['SBP', 'DBP', 'SpO2']:
            # These decrease during sepsis
            change = np.random.uniform(-2.0, -0.5)
        else:
            change = np.random.uniform(-1, 1)
    else:
        # Normal variation
        change = np.random.uniform(-0.5, 0.5)
    
    new_value = current_value + change
    
    # Keep within reasonable bounds
    if vital_name == 'HR':
        new_value = np.clip(new_value, 40, 180)
    elif vital_name in ['SBP', 'DBP']:
        new_value = np.clip(new_value, 50, 200)
    elif vital_name == 'SpO2':
        new_value = np.clip(new_value, 70, 100)
    elif vital_name == 'Resp':
        new_value = np.clip(new_value, 8, 40)
    elif vital_name == 'Temp':
        new_value = np.clip(new_value, 35.0, 41.0)
    
    return new_value

def calculate_risk_score(vitals):
    """Calculate risk score based on vital signs"""
    risk = 0.0
    
    # Heart Rate
    if vitals['HR'] > 100 or vitals['HR'] < 60:
        risk += 0.15
    if vitals['HR'] > 120:
        risk += 0.15
    
    # Blood Pressure
    if vitals['SBP'] < 90:
        risk += 0.25
    if vitals['SBP'] < 80:
        risk += 0.20
    
    # SpO2
    if vitals['SpO2'] < 95:
        risk += 0.15
    if vitals['SpO2'] < 90:
        risk += 0.20
    
    # Respiratory Rate
    if vitals['Resp'] > 20 or vitals['Resp'] < 12:
        risk += 0.10
    if vitals['Resp'] > 24:
        risk += 0.15
    
    # Temperature
    if vitals['Temp'] > 38.0 or vitals['Temp'] < 36.0:
        risk += 0.15
    if vitals['Temp'] > 39.0:
        risk += 0.15
    
    return min(risk, 1.0)

def get_alert_level(risk_score):
    """Get alert level based on risk score"""
    if risk_score >= 0.8:
        return "CRITICAL", "🔴"
    elif risk_score >= 0.6:
        return "HIGH", "🟠"
    elif risk_score >= 0.4:
        return "MODERATE", "🟡"
    else:
        return "NORMAL", "🟢"

def update_vitals(deterioration_mode=False):
    """Update all vital signs"""
    for vital in st.session_state.current_vitals.keys():
        st.session_state.current_vitals[vital] = simulate_vital_change(
            st.session_state.current_vitals[vital],
            vital,
            deterioration_mode
        )
    
    # Update risk score
    st.session_state.risk_score = calculate_risk_score(st.session_state.current_vitals)
    
    # Add to history
    current_time = datetime.now()
    st.session_state.data_history['time'].append(current_time)
    for vital, value in st.session_state.current_vitals.items():
        st.session_state.data_history[vital].append(value)
    
    # Keep only last 50 data points
    if len(st.session_state.data_history['time']) > 50:
        for key in st.session_state.data_history.keys():
            st.session_state.data_history[key] = st.session_state.data_history[key][-50:]
    
    # Add alert if critical
    alert_level, _ = get_alert_level(st.session_state.risk_score)
    if alert_level in ["CRITICAL", "HIGH"]:
        st.session_state.alert_history.append({
            'time': current_time,
            'level': alert_level,
            'risk_score': st.session_state.risk_score
        })
        # Keep only last 10 alerts
        st.session_state.alert_history = st.session_state.alert_history[-10:]

def create_vitals_chart():
    """Create real-time vitals chart"""
    if len(st.session_state.data_history['time']) == 0:
        return None
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Heart Rate', 'Blood Pressure', 'SpO2', 'Respiratory Rate', 'Temperature', 'Risk Score'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    times = st.session_state.data_history['time']
    
    # Heart Rate
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['HR'],
                   mode='lines', name='HR', line=dict(color='red', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=100, line_dash="dash", line_color="orange", row=1, col=1)
    
    # Blood Pressure
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['SBP'],
                   mode='lines', name='SBP', line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['DBP'],
                   mode='lines', name='DBP', line=dict(color='lightblue', width=2)),
        row=1, col=2
    )
    fig.add_hline(y=90, line_dash="dash", line_color="orange", row=1, col=2)
    
    # SpO2
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['SpO2'],
                   mode='lines', name='SpO2', line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=95, line_dash="dash", line_color="orange", row=2, col=1)
    
    # Respiratory Rate
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['Resp'],
                   mode='lines', name='Resp', line=dict(color='purple', width=2)),
        row=2, col=2
    )
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=2)
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=times, y=st.session_state.data_history['Temp'],
                   mode='lines', name='Temp', line=dict(color='orange', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=38.0, line_dash="dash", line_color="red", row=3, col=1)
    
    # Risk Score
    risk_history = [calculate_risk_score({
        'HR': st.session_state.data_history['HR'][i],
        'SBP': st.session_state.data_history['SBP'][i],
        'DBP': st.session_state.data_history['DBP'][i],
        'SpO2': st.session_state.data_history['SpO2'][i],
        'Resp': st.session_state.data_history['Resp'][i],
        'Temp': st.session_state.data_history['Temp'][i]
    }) for i in range(len(times))]
    
    fig.add_trace(
        go.Scatter(x=times, y=risk_history,
                   mode='lines', name='Risk', line=dict(color='red', width=3),
                   fill='tozeroy'),
        row=3, col=2
    )
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=3, col=2)
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="bpm", row=1, col=1)
    fig.update_yaxes(title_text="mmHg", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="/min", row=2, col=2)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_yaxes(title_text="Risk", row=3, col=2)
    
    return fig

# Main UI
st.title("🏥 Real-Time ICU Patient Monitor")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Monitoring Controls")
    
    patient_id = st.text_input("Patient ID", value="ICU-001")
    
    st.markdown("---")
    st.subheader("Simulation Mode")
    
    simulation_mode = st.radio(
        "Select Mode",
        ["Normal Variation", "Deterioration", "Manual Control"],
        help="Choose how vitals change over time"
    )
    
    if simulation_mode == "Manual Control":
        st.markdown("### Manual Vital Adjustment")
        manual_hr = st.slider("Heart Rate", 40, 180, int(st.session_state.current_vitals['HR']))
        manual_sbp = st.slider("Systolic BP", 50, 200, int(st.session_state.current_vitals['SBP']))
        manual_spo2 = st.slider("SpO2", 70, 100, int(st.session_state.current_vitals['SpO2']))
        manual_resp = st.slider("Resp Rate", 8, 40, int(st.session_state.current_vitals['Resp']))
        manual_temp = st.slider("Temperature", 35.0, 41.0, float(st.session_state.current_vitals['Temp']), 0.1)
        
        if st.button("Apply Manual Values"):
            st.session_state.current_vitals['HR'] = manual_hr
            st.session_state.current_vitals['SBP'] = manual_sbp
            st.session_state.current_vitals['DBP'] = manual_sbp - 40
            st.session_state.current_vitals['SpO2'] = manual_spo2
            st.session_state.current_vitals['Resp'] = manual_resp
            st.session_state.current_vitals['Temp'] = manual_temp
            st.session_state.risk_score = calculate_risk_score(st.session_state.current_vitals)
    
    st.markdown("---")
    st.subheader("Monitoring Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", width="stretch"):
            st.session_state.monitoring = True
    with col2:
        if st.button("⏸️ Pause", width="stretch"):
            st.session_state.monitoring = False
    
    if st.button("🔄 Reset", width="stretch"):
        st.session_state.monitoring = False
        st.session_state.data_history = {key: [] for key in st.session_state.data_history.keys()}
        st.session_state.current_vitals = {
            'HR': 75, 'SBP': 120, 'DBP': 80,
            'SpO2': 98, 'Resp': 16, 'Temp': 37.0
        }
        st.session_state.risk_score = 0.25
        st.session_state.alert_history = []
        st.session_state.time_elapsed = 0
        st.rerun()
    
    st.markdown("---")
    st.subheader("Update Speed")
    update_interval = st.slider("Seconds between updates", 0.5, 5.0, 1.0, 0.5)
    
    st.markdown("---")
    st.info(f"**Status:** {'🟢 Monitoring' if st.session_state.monitoring else '🔴 Paused'}")
    st.info(f"**Time Elapsed:** {st.session_state.time_elapsed}s")

# Main display area
alert_level, alert_icon = get_alert_level(st.session_state.risk_score)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Risk Score",
        f"{st.session_state.risk_score:.1%}",
        delta=f"{alert_level}",
        delta_color="inverse" if alert_level in ["CRITICAL", "HIGH"] else "normal"
    )

with col2:
    st.metric("Heart Rate", f"{st.session_state.current_vitals['HR']:.0f} bpm")

with col3:
    st.metric("Blood Pressure", f"{st.session_state.current_vitals['SBP']:.0f}/{st.session_state.current_vitals['DBP']:.0f}")

with col4:
    st.metric("SpO2", f"{st.session_state.current_vitals['SpO2']:.0f}%")

# Alert banner
if alert_level == "CRITICAL":
    st.markdown(f'<div class="alert-critical">{alert_icon} CRITICAL ALERT - Immediate Intervention Required!</div>', unsafe_allow_html=True)
elif alert_level == "HIGH":
    st.markdown(f'<div class="alert-high">{alert_icon} HIGH ALERT - Close Monitoring Required</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-normal">{alert_icon} Patient Status: {alert_level}</div>', unsafe_allow_html=True)

st.markdown("---")

# Charts and vitals
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Real-Time Vital Signs")
    chart_placeholder = st.empty()
    
    if len(st.session_state.data_history['time']) > 0:
        fig = create_vitals_chart()
        if fig:
            chart_placeholder.plotly_chart(fig, width="stretch")
    else:
        chart_placeholder.info("Start monitoring to see real-time data...")

with col_right:
    st.subheader("📊 Current Vitals")
    
    vitals_display = st.container()
    with vitals_display:
        st.markdown(f"**Heart Rate:** {st.session_state.current_vitals['HR']:.0f} bpm")
        st.progress(min(st.session_state.current_vitals['HR'] / 180, 1.0))
        
        st.markdown(f"**Systolic BP:** {st.session_state.current_vitals['SBP']:.0f} mmHg")
        st.progress(min(st.session_state.current_vitals['SBP'] / 200, 1.0))
        
        st.markdown(f"**SpO2:** {st.session_state.current_vitals['SpO2']:.0f}%")
        st.progress(st.session_state.current_vitals['SpO2'] / 100)
        
        st.markdown(f"**Resp Rate:** {st.session_state.current_vitals['Resp']:.0f} /min")
        st.progress(min(st.session_state.current_vitals['Resp'] / 40, 1.0))
        
        st.markdown(f"**Temperature:** {st.session_state.current_vitals['Temp']:.1f} °C")
        st.progress(min((st.session_state.current_vitals['Temp'] - 35) / 6, 1.0))

st.markdown("---")

# Recommendations and alerts
col_rec, col_alerts = st.columns(2)

with col_rec:
    st.subheader("💊 Recommended Actions")
    
    if HAS_RAG and st.session_state.risk_score > 0.4:
        # Get recommendations
        normalized_vitals = {
            'HR': (st.session_state.current_vitals['HR'] - 60) / 100,
            'SBP': (st.session_state.current_vitals['SBP'] - 70) / 100,
            'SpO2': (st.session_state.current_vitals['SpO2'] - 70) / 30
        }
        
        top_features = []
        if st.session_state.current_vitals['HR'] > 100:
            top_features.append('HR')
        if st.session_state.current_vitals['SBP'] < 90:
            top_features.append('SBP')
        if st.session_state.current_vitals['SpO2'] < 95:
            top_features.append('O2Sat')
        
        if not top_features:
            top_features = ['HR', 'SBP']
        
        rag = ClinicalRAG()
        recommendation = rag.get_recommendation(
            risk_score=st.session_state.risk_score,
            top_features=top_features[:3],
            patient_id=patient_id
        )
        
        for i, action in enumerate(recommendation.recommended_actions[:5], 1):
            st.markdown(f"**{i}.** {action}")
    else:
        # Fallback recommendations
        if alert_level == "CRITICAL":
            st.markdown("**1.** Activate rapid response team")
            st.markdown("**2.** Start IV fluid resuscitation")
            st.markdown("**3.** Obtain blood cultures")
            st.markdown("**4.** Administer broad-spectrum antibiotics")
            st.markdown("**5.** Monitor vitals every 5 minutes")
        elif alert_level == "HIGH":
            st.markdown("**1.** Notify attending physician")
            st.markdown("**2.** Increase monitoring frequency")
            st.markdown("**3.** Review recent lab results")
            st.markdown("**4.** Prepare for possible escalation")
        else:
            st.markdown("**1.** Continue routine monitoring")
            st.markdown("**2.** Standard vital sign checks")
            st.markdown("**3.** Document any changes")

with col_alerts:
    st.subheader("🚨 Alert History")
    
    if st.session_state.alert_history:
        for alert in reversed(st.session_state.alert_history[-5:]):
            time_str = alert['time'].strftime('%H:%M:%S')
            level_icon = "🔴" if alert['level'] == "CRITICAL" else "🟠"
            st.warning(f"{level_icon} **{alert['level']}** at {time_str} - Risk: {alert['risk_score']:.1%}")
    else:
        st.info("No alerts recorded")

# Auto-update logic
if st.session_state.monitoring:
    time.sleep(update_interval)
    
    deterioration = (simulation_mode == "Deterioration")
    if simulation_mode != "Manual Control":
        update_vitals(deterioration)
    
    st.session_state.time_elapsed += int(update_interval)
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Patient: {patient_id} | Mode: {simulation_mode}")
