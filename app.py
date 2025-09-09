"""
Diabetes AI System - Streamlit Web Interface
"""

import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from diabetes_ai_system import DiabetesAISystem, MealAnalysisRequest, ComprehensiveReport
from agents.diabetes_advisor_agent import DiabetesProfile
import config

# Page configuration
st.set_page_config(
    page_title="Diabetes AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high {
        border-left-color: #d62728 !important;
        background-color: #fee;
    }
    .risk-medium {
        border-left-color: #ff7f0e !important;
        background-color: #ffd;
    }
    .risk-low {
        border-left-color: #2ca02c !important;
        background-color: #efe;
    }
    .insight-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0277bd;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'patient_profile_data' not in st.session_state:
        st.session_state.patient_profile_data = None
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None

def create_patient_profile_form():
    """Create patient profile form in sidebar"""
    st.sidebar.header("👤 Patient Profile")
    
    with st.sidebar.form("patient_profile_form"):
        diabetes_type = st.selectbox(
            "Diabetes Type",
            ["Type 1", "Type 2", "Gestational", "Prediabetes"]
        )
        
        medications = st.text_area(
            "Current Medications",
            placeholder="e.g., Metformin, Insulin"
        )
        
        activity_level = st.selectbox(
            "Activity Level",
            ["sedentary", "moderately_active", "very_active"]
        )
        
        target_fasting = st.number_input(
            "Target Fasting Blood Sugar (mg/dL)",
            min_value=70,
            max_value=200,
            value=100
        )
        
        target_post_meal = st.number_input(
            "Target Post-Meal Blood Sugar (mg/dL)",
            min_value=100,
            max_value=300,
            value=140
        )
        
        dietary_restrictions = st.text_area(
            "Dietary Restrictions",
            placeholder="e.g., Vegetarian, Gluten-free"
        )
        
        last_hba1c = st.number_input(
            "Last HbA1c (%)",
            min_value=4.0,
            max_value=15.0,
            value=7.0,
            step=0.1
        )
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            profile = DiabetesProfile(
                diabetes_type=diabetes_type,
                current_medication=medications.split(',') if medications else [],
                target_blood_sugar={
                    "fasting": target_fasting,
                    "post_meal": target_post_meal
                },
                activity_level=activity_level,
                dietary_restrictions=dietary_restrictions.split(',') if dietary_restrictions else [],
                last_hba1c=last_hba1c if last_hba1c > 0 else None
            )
            st.session_state.patient_profile_data = profile
            st.sidebar.success("Profile saved!")

def display_food_recognition_results(food_result):
    """Display food recognition results"""
    st.subheader("🔍 Food Recognition Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Meal Type:** {food_result.meal_type.title()}")
        st.write(f"**Total Items Identified:** {food_result.total_items}")
        
        if food_result.foods:
            food_data = []
            for food in food_result.foods:
                food_data.append({
                    "Food Item": food.name,
                    "Quantity": food.quantity,
                    "Confidence": f"{food.confidence:.1%}",
                    "Description": food.description[:50] + "..." if len(food.description) > 50 else food.description
                })
            
            df = pd.DataFrame(food_data)
            st.dataframe(df, width='stretch')
    
    with col2:
        # Create confidence chart
        if food_result.foods:
            fig, ax = plt.subplots(figsize=(6, 4))
            confidences = [food.confidence for food in food_result.foods]
            food_names = [food.name[:15] + "..." if len(food.name) > 15 else food.name for food in food_result.foods]
            
            bars = ax.barh(food_names, confidences)
            ax.set_xlabel("Confidence Score")
            ax.set_title("Recognition Confidence")
            ax.set_xlim(0, 1)
            
            # Color bars based on confidence
            for i, bar in enumerate(bars):
                if confidences[i] >= 0.8:
                    bar.set_color('#2ca02c')
                elif confidences[i] >= 0.6:
                    bar.set_color('#ff7f0e')
                else:
                    bar.set_color('#d62728')
            
            plt.tight_layout()
            st.pyplot(fig)

def display_nutrition_analysis(nutrition_result):
    """Display nutrition analysis results"""
    st.subheader("📊 Nutrition Analysis")
    
    # Main nutrition metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Calories",
            value=f"{nutrition_result.total_calories:.0f}",
            delta="kcal"
        )
    
    with col2:
        st.metric(
            label="Total Carbs",
            value=f"{nutrition_result.total_carbs_g:.1f}g",
            delta=f"{(nutrition_result.total_carbs_g/45)*100:.0f}% of 45g limit" if nutrition_result.total_carbs_g > 45 else None
        )
    
    with col3:
        st.metric(
            label="Total Sugars",
            value=f"{nutrition_result.total_sugars_g:.1f}g",
            delta=f"{(nutrition_result.total_sugars_g/25)*100:.0f}% of daily limit" if nutrition_result.total_sugars_g > 25 else None
        )
    
    with col4:
        risk_color = "🔴" if nutrition_result.diabetes_risk_score >= 7 else "🟡" if nutrition_result.diabetes_risk_score >= 5 else "🟢"
        st.metric(
            label="Diabetes Risk",
            value=f"{risk_color} {nutrition_result.diabetes_risk_score}/10"
        )
    
    # Detailed nutrition breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Macronutrient pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate calories from macronutrients
        carb_cal = nutrition_result.total_carbs_g * 4
        protein_cal = nutrition_result.total_protein_g * 4
        fat_cal = nutrition_result.total_fat_g * 9
        
        # Handle NaN values and ensure we have valid data
        sizes = [max(0, carb_cal or 0), max(0, protein_cal or 0), max(0, fat_cal or 0)]
        
        # Only create pie chart if we have valid data
        if sum(sizes) > 0:
            labels = ['Carbohydrates', 'Protein', 'Fat']
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Macronutrient Distribution (by calories)')
        else:
            ax.text(0.5, 0.5, 'No nutrition data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Macronutrient Distribution (by calories)')
        st.pyplot(fig)
    
    with col2:
        # Glycemic impact
        if nutrition_result.total_glycemic_load:
            gl = nutrition_result.total_glycemic_load
            gl_level = "High" if gl >= 20 else "Medium" if gl >= 10 else "Low"
            gl_color = "#d62728" if gl >= 20 else "#ff7f0e" if gl >= 10 else "#2ca02c"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Glycemic Load Assessment</h4>
                <p><strong>Glycemic Load:</strong> <span style="color: {gl_color}">{gl:.1f} ({gl_level})</span></p>
                <p><strong>Average GI:</strong> {nutrition_result.average_glycemic_index:.1f} (estimated)</p>
                <p><small>
                • Low GL (&lt;10): Minimal blood sugar impact<br>
                • Medium GL (10-19): Moderate impact<br>
                • High GL (≥20): Significant impact
                </small></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Individual food nutrition
        st.subheader("Individual Food Items")
        for food in nutrition_result.individual_foods:
            with st.expander(f"{food.food_name} ({food.serving_size})"):
                food_cal = food.nutrition_per_serving.calories_per_100g * food.estimated_weight_g / 100
                food_carbs = food.nutrition_per_serving.total_carbs_g * food.estimated_weight_g / 100
                
                st.write(f"**Estimated Weight:** {food.estimated_weight_g:.0f}g")
                st.write(f"**Calories:** {food_cal:.0f}")
                st.write(f"**Carbohydrates:** {food_carbs:.1f}g")
                st.write(f"**Diabetes Impact:** {food.diabetes_impact}")

def display_diabetes_advisory(advisory_result):
    """Display diabetes advisory results"""
    st.subheader("🩺 Diabetes Management Advisory")
    
    # Blood sugar prediction
    if advisory_result.blood_sugar_prediction:
        prediction = advisory_result.blood_sugar_prediction
        
        risk_class = "risk-high" if prediction.risk_level == "High" else "risk-medium" if prediction.risk_level == "Moderate" else "risk-low"
        
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h4>🩸 Blood Sugar Prediction</h4>
            <p><strong>Peak Time:</strong> {prediction.predicted_peak_time}</p>
            <p><strong>Expected Range:</strong> {prediction.predicted_peak_range}</p>
            <p><strong>Duration Elevated:</strong> {prediction.duration_elevated}</p>
            <p><strong>Risk Level:</strong> {prediction.risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Monitoring advice
        st.subheader("🔍 Monitoring Recommendations")
        for advice in prediction.monitoring_advice:
            st.write(f"• {advice}")
    
    # Personalized advice
    if advisory_result.personalized_advice:
        advice = advisory_result.personalized_advice
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Immediate Actions")
            for action in advice.immediate_actions:
                st.write(f"• {action}")
            
            st.subheader("🍽️ Meal Modifications")
            for mod in advice.meal_modifications:
                st.write(f"• {mod}")
        
        with col2:
            st.subheader("📋 Future Planning")
            for plan in advice.future_planning:
                st.write(f"• {plan}")
            
            st.subheader("🚨 Emergency Protocols")
            for protocol in advice.emergency_protocols:
                st.write(f"• {protocol}")
    
    # Healthcare alerts
    if advisory_result.healthcare_alerts:
        st.subheader("🏥 Healthcare Provider Alerts")
        for alert in advisory_result.healthcare_alerts:
            alert_type = "error" if "error" in alert.lower() else "warning" if any(word in alert.lower() for word in ["high", "risk", "concern"]) else "info"
            st.write(f":{alert_type}: {alert}")

def display_comprehensive_summary(report: ComprehensiveReport):
    """Display comprehensive summary"""
    st.subheader("📋 Summary & Key Insights")
    
    # Summary score
    score_color = "🔴" if report.summary_score <= 4 else "🟡" if report.summary_score <= 7 else "🟢"
    st.metric(
        label="Overall Meal Score",
        value=f"{score_color} {report.summary_score}/10",
        help="Higher scores indicate better suitability for diabetes management"
    )
    
    # Key insights
    st.subheader("💡 Key Insights")
    for insight in report.key_insights:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)
    
    # Action items
    st.subheader("✅ Action Items")
    for i, action in enumerate(report.action_items, 1):
        st.write(f"{i}. {action}")
    
    # Daily tracking notes
    if report.diabetes_advisory.daily_tracking_notes:
        st.subheader("📝 Daily Tracking Notes")
        for note in report.diabetes_advisory.daily_tracking_notes:
            st.write(f"• {note}")

def display_analysis_history():
    """Display analysis history"""
    if st.session_state.analysis_history:
        st.subheader("📊 Analysis History")
        
        history_data = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            history_data.append({
                "Time": analysis["timestamp"][:19].replace("T", " "),
                "Foods": len(analysis["report"].food_recognition.foods),
                "Calories": f"{analysis['report'].nutrition_analysis.total_calories:.0f}",
                "Risk Score": f"{analysis['report'].nutrition_analysis.diabetes_risk_score}/10",
                "Meal Score": f"{analysis['report'].summary_score}/10"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, width='stretch')
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered food analysis for diabetes management*")
    
    # Sidebar for patient profile
    create_patient_profile_form()
    
    # API Key check
    if not config.OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Please set your OPENAI_API_KEY environment variable.")
        st.info("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Analyze Meal", "📊 Current Report", "📈 History"])
    
    with tab1:
        st.header("Upload Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image of your meal for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width='stretch')
            
            with col2:
                st.write("**Image Details:**")
                st.write(f"📁 Filename: {uploaded_file.name}")
                st.write(f"📏 Size: {image.size}")
                st.write(f"💾 File size: {len(uploaded_file.getvalue())/1024:.1f} KB")
                
                # Optional meal notes
                meal_notes = st.text_area(
                    "Additional Notes (optional)",
                    placeholder="e.g., 'Ate at lunch time', 'Feeling hungry', 'Post-workout meal'"
                )
                
                # Analysis button
                if st.button("🔍 Analyze Meal", type="primary"):
                    with st.spinner("Analyzing your meal... This may take a few moments."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_path = tmp_file.name
                            
                            # Create analysis request
                            request = MealAnalysisRequest(
                                image_path=temp_path,
                                meal_time=datetime.now().isoformat(),
                                patient_notes=meal_notes if meal_notes else None,
                                patient_profile=st.session_state.patient_profile_data
                            )
                            
                            # Run analysis
                            system = DiabetesAISystem()
                            report = system.analyze_meal(request)
                            
                            # Store in session state
                            st.session_state.current_report = report
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                "timestamp": report.timestamp,
                                "report": report,
                                "image_name": uploaded_file.name
                            })
                            
                            # Clean up temp file
                            os.unlink(temp_path)
                            
                            st.success("✅ Analysis complete! Check the 'Current Report' tab for detailed results.")
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            st.info("Please try again with a different image or check your internet connection.")
    
    with tab2:
        if st.session_state.current_report:
            report = st.session_state.current_report
            
            # Display all analysis results
            display_food_recognition_results(report.food_recognition)
            st.divider()
            display_nutrition_analysis(report.nutrition_analysis)
            st.divider()
            display_diabetes_advisory(report.diabetes_advisory)
            st.divider()
            display_comprehensive_summary(report)
            
            # Download report
            if st.button("💾 Download Report as JSON"):
                report_json = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"diabetes_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("📸 No analysis report available. Please analyze a meal first.")
    
    with tab3:
        display_analysis_history()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>
        ⚠️ <strong>Medical Disclaimer:</strong> This AI assistant is for educational purposes only. 
        Always consult with your healthcare provider for medical advice and diabetes management decisions.
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
