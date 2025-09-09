"""
Mobile-optimized Diabetes AI Assistant
"""

import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import base64

from diabetes_ai_system import DiabetesAISystem, MealAnalysisRequest, ComprehensiveReport
from agents.diabetes_advisor_agent import DiabetesProfile
import config
from mobile_config import MOBILE_CONFIG, CAMERA_CONFIG, MOBILE_UI

# Configure for mobile
st.set_page_config(**MOBILE_CONFIG)

# Mobile-specific CSS
st.markdown("""
<style>
    /* Mobile-optimized styles */
    .main-container {
        padding: 0.5rem;
        max-width: 100vw;
    }
    
    .mobile-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .camera-button {
        background: #ff6b35;
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255,107,53,0.3);
    }
    
    .quick-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .mobile-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .food-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    .risk-indicator {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
    
    .risk-low { background-color: #2ca02c; }
    .risk-medium { background-color: #ff7f0e; }
    .risk-high { background-color: #d62728; }
    
    /* Hide desktop elements on mobile */
    @media (max-width: 768px) {
        .desktop-only { display: none !important; }
        .sidebar .sidebar-content { display: none; }
    }
    
    /* Touch-friendly buttons */
    .stButton > button {
        height: 3rem;
        font-size: 1.1rem;
        border-radius: 25px;
    }
    
    /* Mobile-friendly file uploader */
    .uploadedFile {
        border-radius: 15px;
        border: 2px dashed #1f77b4;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_mobile_session():
    """Initialize mobile-specific session state"""
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = True
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'patient_profile_data' not in st.session_state:
        st.session_state.patient_profile_data = None
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None

def mobile_header():
    """Create mobile-optimized header"""
    st.markdown("""
    <div class="mobile-header">
        <h1>🩺 Diabetes AI</h1>
        <p>Snap, Analyze, Manage</p>
    </div>
    """, unsafe_allow_html=True)

def camera_interface():
    """Mobile camera interface for food photos"""
    st.markdown("### 📸 Take a Photo of Your Meal")
    
    # Camera input (works on mobile browsers)
    uploaded_file = st.file_uploader(
        "📱 Tap to take photo or upload image",
        type=['jpg', 'jpeg', 'png'],
        help="Take a clear photo of your meal",
        key="mobile_camera"
    )
    
    if uploaded_file is not None:
        return uploaded_file
    return None

def display_mobile_food_results(food_result):
    """Mobile-optimized food recognition display"""
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("#### 🍽️ Identified Foods")
    
    if food_result.foods:
        for food in food_result.foods:
            confidence_color = "#2ca02c" if food.confidence > 0.8 else "#ff7f0e" if food.confidence > 0.6 else "#d62728"
            
            st.markdown(f"""
            <div class="food-item">
                <div>
                    <strong>{food.name}</strong><br>
                    <small>{food.quantity}</small>
                </div>
                <div style="color: {confidence_color}; font-weight: bold;">
                    {food.confidence:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f"**Meal Type:** {food_result.meal_type.title()}")
    st.markdown('</div>', unsafe_allow_html=True)

def display_mobile_nutrition_summary(nutrition_result):
    """Mobile-optimized nutrition display"""
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Nutrition Summary")
    
    # Quick stats grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="quick-stats">
            <h3>{nutrition_result.total_calories:.0f}</h3>
            <p>Calories</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="quick-stats">
            <h3>{nutrition_result.total_carbs_g:.1f}g</h3>
            <p>Carbs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="quick-stats">
            <h3>{nutrition_result.total_sugars_g:.1f}g</h3>
            <p>Sugars</p>
        </div>
        """, unsafe_allow_html=True)
        
        risk_class = "risk-high" if nutrition_result.diabetes_risk_score >= 7 else "risk-medium" if nutrition_result.diabetes_risk_score >= 5 else "risk-low"
        st.markdown(f"""
        <div class="quick-stats">
            <h3><span class="risk-indicator {risk_class}"></span>{nutrition_result.diabetes_risk_score}/10</h3>
            <p>Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_mobile_recommendations(advisory_result):
    """Mobile-optimized recommendations"""
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.markdown("#### 💡 Recommendations")
    
    if advisory_result.personalized_advice and advisory_result.personalized_advice.immediate_actions:
        st.markdown("**🚨 Immediate Actions:**")
        for action in advisory_result.personalized_advice.immediate_actions[:3]:  # Show only top 3
            st.markdown(f"• {action}")
    
    if advisory_result.blood_sugar_prediction:
        prediction = advisory_result.blood_sugar_prediction
        st.markdown(f"""
        **🩸 Blood Sugar Impact:**
        - Peak expected: {prediction.predicted_peak_time}
        - Risk level: {prediction.risk_level}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def quick_patient_setup():
    """Quick mobile patient setup"""
    with st.expander("⚙️ Quick Setup"):
        diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2", "Gestational"])
        activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
        
        if st.button("Save Profile"):
            profile = DiabetesProfile(
                diabetes_type=diabetes_type,
                current_medication=["Unknown"],
                target_blood_sugar={"fasting": 100, "post_meal": 140},
                activity_level=activity_level.lower(),
                dietary_restrictions=[],
                last_hba1c=None
            )
            st.session_state.patient_profile_data = profile
            st.success("Profile saved!")

def analysis_history_mobile():
    """Mobile-friendly history view"""
    if st.session_state.analysis_history:
        st.markdown("#### 📈 Recent Analysis")
        
        for i, analysis in enumerate(st.session_state.analysis_history[-3:]):  # Show last 3
            with st.expander(f"Meal {i+1} - {analysis['timestamp'][:10]}"):
                report = analysis['report']
                st.write(f"**Foods:** {len(report.food_recognition.foods)} items")
                st.write(f"**Calories:** {report.nutrition_analysis.total_calories:.0f}")
                st.write(f"**Risk Score:** {report.nutrition_analysis.diabetes_risk_score}/10")

def main():
    """Main mobile application"""
    initialize_mobile_session()
    
    # API Key check
    if not config.OPENAI_API_KEY:
        st.error("⚠️ API key required. Please configure your OpenAI API key.")
        return
    
    # Mobile header
    mobile_header()
    
    # Quick patient setup
    quick_patient_setup()
    
    # Main camera interface
    uploaded_file = camera_interface()
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Meal", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Meal", key="analyze_mobile", help="Tap to start AI analysis"):
            with st.spinner("🤖 AI is analyzing your meal..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Create analysis request
                    request = MealAnalysisRequest(
                        image_path=temp_path,
                        meal_time=datetime.now().isoformat(),
                        patient_notes=None,
                        patient_profile=st.session_state.patient_profile_data
                    )
                    
                    # Run analysis
                    system = DiabetesAISystem()
                    report = system.analyze_meal(request)
                    
                    # Store results
                    st.session_state.current_report = report
                    st.session_state.analysis_history.append({
                        "timestamp": report.timestamp,
                        "report": report,
                        "image_name": uploaded_file.name
                    })
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results if available
    if st.session_state.current_report:
        report = st.session_state.current_report
        
        # Mobile-optimized displays
        display_mobile_food_results(report.food_recognition)
        display_mobile_nutrition_summary(report.nutrition_analysis)
        display_mobile_recommendations(report.diabetes_advisory)
        
        # Quick share button
        if st.button("📤 Share Results"):
            # Create shareable summary
            summary = f"""
🩺 Diabetes AI Analysis
📅 {report.timestamp[:10]}
🍽️ {len(report.food_recognition.foods)} foods identified
📊 {report.nutrition_analysis.total_calories:.0f} calories, {report.nutrition_analysis.total_carbs_g:.1f}g carbs
⚠️ Risk Score: {report.nutrition_analysis.diabetes_risk_score}/10
            """
            st.text_area("Share this summary:", summary, height=150)
    
    # History section
    analysis_history_mobile()
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>
        🩺 Diabetes AI Assistant - For educational purposes only<br>
        Always consult your healthcare provider
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
