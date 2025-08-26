import streamlit as st
import joblib
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div { padding: 2rem 1rem; }
    .stApp > header { background-color: transparent; }
    .title-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content { background-color: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

# Load the model safely
@st.cache_resource
def load_model():
    model_path = "Diabetes_prediction/log_model.pkl"  # ✅ Corrected filename
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please upload it to the repository.")
        return None
    return joblib.load(model_path)

model = load_model()

# Title with custom styling
st.markdown("""
<div class="title-container">
    <h1>🩺 Advanced Diabetes Prediction System</h1>
    <p style="font-size: 1.2rem; margin: 0;">AI-Powered Health Assessment Tool</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional information
st.sidebar.title("📊 About This Tool")
st.sidebar.markdown("""
This diabetes prediction tool uses machine learning to assess diabetes risk based on key health indicators.

**Key Features:**
- 🤖 AI-powered predictions
- 📈 Real-time probability scoring
- 📋 Comprehensive health metrics
- 🎯 Instant results

**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
""")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📋 Patient Information")
    
    with st.expander("👤 Personal Details", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, help="Number of times pregnant")
            age = st.number_input("Age", min_value=0, max_value=120, step=1, help="Age in years")
        with col_b:
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, step=0.1, help="Plasma glucose concentration")
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1, help="Body Mass Index")
    
    with st.expander("🩸 Clinical Measurements", expanded=True):
        col_c, col_d = st.columns(2)
        with col_c:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, step=0.1, help="Diastolic blood pressure")
            insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=1000.0, step=0.1, help="2-Hour serum insulin")
        with col_d:
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, step=0.1, help="Triceps skin fold thickness")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.001, help="Genetic predisposition score")

with col2:
    st.markdown("## 📊 Health Indicators")
    st.markdown("""
    <div class="info-box">
        <h4>📈 Normal Ranges</h4>
        <ul>
            <li><strong>Glucose:</strong> 70-100 mg/dL (fasting)</li>
            <li><strong>Blood Pressure:</strong> <90 mmHg (diastolic)</li>
            <li><strong>BMI:</strong> 18.5-24.9 kg/m²</li>
            <li><strong>Insulin:</strong> 2.6-24.9 μU/ml</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Risk Factors</h4>
        <ul>
            <li>High glucose levels</li>
            <li>Family history (high DPF)</li>
            <li>Obesity (BMI >30)</li>
            <li>Age >45 years</li>
            <li>High blood pressure</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction section
st.markdown("---")
st.markdown("## 🔮 Prediction Results")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 Analyze Diabetes Risk", use_container_width=True, type="primary")

if predict_button:
    if model is None:
        st.markdown("""
        <div class="danger-box">
            <h4>⚠️ Model Not Available</h4>
            <p>The prediction model could not be loaded. Please check the file path and try again.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_features = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])
        
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1] * 100
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>🚨 High Risk Detected</h3>
                    <h2 style="color: #d32f2f; margin: 0;">DIABETIC</h2>
                    <p style="font-size: 1.1rem;">Risk Probability: <strong>{probability:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="warning-box">
                    <h4>🏥 Recommended Actions</h4>
                    <ul>
                        <li>Consult with a healthcare provider immediately</li>
                        <li>Get comprehensive diabetes screening</li>
                        <li>Monitor blood sugar levels regularly</li>
                        <li>Consider lifestyle modifications</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Low Risk Detected</h3>
                    <h2 style="color: #2e7d32; margin: 0;">NON-DIABETIC</h2>
                    <p style="font-size: 1.1rem;">Risk Probability: <strong>{probability:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <h4>🌟 Preventive Measures</h4>
                    <ul>
                        <li>Maintain a healthy diet</li>
                        <li>Exercise regularly</li>
                        <li>Monitor your health annually</li>
                        <li>Keep BMI in normal range</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability > 50 else "darkgreen"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📄 Input Summary")
            summary_data = {
                "Parameter": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", 
                             "Insulin", "BMI", "DPF", "Age"],
                "Value": [pregnancies, f"{glucose:.1f}", f"{blood_pressure:.1f}", 
                         f"{skin_thickness:.1f}", f"{insulin:.1f}", f"{bmi:.1f}", 
                         f"{dpf:.3f}", age]
            }
            for param, val in zip(summary_data["Parameter"], summary_data["Value"]):
                st.write(f"**{param}:** {val}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>📊 Prediction generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><em>This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)

