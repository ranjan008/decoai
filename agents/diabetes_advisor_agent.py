from typing import List, Dict, Optional
from datetime import datetime, timedelta
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import config
from agents.nutrition_analysis_agent import MealNutritionSummary

class DiabetesProfile(BaseModel):
    """Patient diabetes profile"""
    diabetes_type: str = Field(description="Type 1, Type 2, or Gestational")
    current_medication: List[str] = Field(description="Current diabetes medications")
    target_blood_sugar: Dict[str, float] = Field(description="Target blood sugar ranges")
    activity_level: str = Field(description="sedentary, moderately_active, very_active")
    dietary_restrictions: List[str] = Field(description="Any dietary restrictions")
    last_hba1c: Optional[float] = Field(description="Most recent HbA1c level")

class MealTiming(BaseModel):
    """Meal timing recommendations"""
    optimal_timing: str = Field(description="Best time to eat this meal")
    spacing_recommendation: str = Field(description="Recommended spacing from other meals")
    medication_timing: List[str] = Field(description="Medication timing advice")
    exercise_timing: str = Field(description="Exercise timing recommendations")

class BloodSugarPrediction(BaseModel):
    """Predicted blood sugar response"""
    predicted_peak_time: str = Field(description="When blood sugar will likely peak")
    predicted_peak_range: str = Field(description="Expected peak blood sugar range")
    duration_elevated: str = Field(description="How long blood sugar may stay elevated")
    risk_level: str = Field(description="Low, Moderate, High, Critical")
    monitoring_advice: List[str] = Field(description="When and how to monitor")

class PersonalizedAdvice(BaseModel):
    """Personalized diabetes management advice"""
    immediate_actions: List[str] = Field(description="Actions to take right now")
    meal_modifications: List[str] = Field(description="How to modify this meal")
    future_planning: List[str] = Field(description="Planning for similar meals")
    emergency_protocols: List[str] = Field(description="What to do if problems arise")
    lifestyle_tips: List[str] = Field(description="Ongoing lifestyle recommendations")

class DiabetesAdvisoryReport(BaseModel):
    """Complete diabetes management report"""
    meal_summary: MealNutritionSummary = Field(description="Nutritional analysis")
    meal_timing: Optional[MealTiming] = Field(default=None, description="Timing recommendations")
    blood_sugar_prediction: Optional[BloodSugarPrediction] = Field(default=None, description="Expected blood sugar response")
    personalized_advice: Optional[PersonalizedAdvice] = Field(default=None, description="Customized recommendations")
    daily_tracking_notes: List[str] = Field(description="Notes for daily tracking")
    healthcare_alerts: List[str] = Field(description="Alerts for healthcare provider")

class BloodSugarPredictionTool(BaseTool):
    """Tool for predicting blood sugar response to meals"""
    name: str = "predict_blood_sugar_response"
    description: str = "Predict blood sugar response based on meal composition and patient profile"
    
    def _run(self, nutrition_summary: str, patient_profile: str) -> str:
        """Predict blood sugar response"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            As an endocrinologist, predict the blood sugar response for:
            
            MEAL NUTRITION:
            {nutrition_summary}
            
            PATIENT PROFILE:
            {patient_profile}
            
            Provide detailed prediction including:
            1. Expected blood sugar peak (time and range)
            2. Duration of elevation
            3. Risk assessment
            4. Monitoring recommendations
            5. Intervention thresholds
            
            Consider:
            - Glycemic index and load
            - Meal composition (protein, fat, fiber effects)
            - Individual metabolic factors
            - Medication timing
            - Activity level
            
            Be specific about timing and values.
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error predicting blood sugar response: {str(e)}"

class MealTimingTool(BaseTool):
    """Tool for optimizing meal timing"""
    name: str = "optimize_meal_timing"
    description: str = "Provide optimal timing recommendations for meals based on diabetes management"
    
    def _run(self, meal_type: str, nutrition_data: str, patient_profile: str, current_time: str) -> str:
        """Optimize meal timing"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            As a diabetes educator, optimize meal timing for:
            
            MEAL TYPE: {meal_type}
            CURRENT TIME: {current_time}
            NUTRITION DATA: {nutrition_data}
            PATIENT PROFILE: {patient_profile}
            
            Provide recommendations for:
            1. Optimal eating time
            2. Spacing from other meals
            3. Medication timing coordination
            4. Pre/post-meal activities
            5. Sleep considerations (if dinner/evening snack)
            
            Consider:
            - Circadian rhythm effects on insulin sensitivity
            - Medication peak times
            - Work/sleep schedule
            - Exercise timing
            - Previous meal timing
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error optimizing meal timing: {str(e)}"

class PersonalizationTool(BaseTool):
    """Tool for generating personalized diabetes advice"""
    name: str = "generate_personalized_advice"
    description: str = "Generate personalized diabetes management advice based on individual profile"
    
    def _run(self, meal_analysis: str, patient_profile: str, risk_factors: str) -> str:
        """Generate personalized advice"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.2
            )
            
            prompt = f"""
            As a certified diabetes educator, provide personalized advice for:
            
            MEAL ANALYSIS: {meal_analysis}
            PATIENT PROFILE: {patient_profile}
            RISK FACTORS: {risk_factors}
            
            Provide comprehensive advice including:
            1. Immediate action items
            2. Meal modification strategies
            3. Future meal planning
            4. Emergency protocols
            5. Lifestyle optimization tips
            6. Healthcare provider communication points
            
            Tailor advice to:
            - Diabetes type and management style
            - Lifestyle and preferences
            - Risk tolerance
            - Previous patterns
            - Medication regimen
            
            Be practical, actionable, and encouraging.
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error generating personalized advice: {str(e)}"

class EmergencyAssessmentTool(BaseTool):
    """Tool for assessing emergency situations"""
    name: str = "assess_emergency_risk"
    description: str = "Assess if meal presents emergency diabetes risk and provide protocols"
    
    def _run(self, nutrition_summary: str, risk_score: int, patient_profile: str) -> str:
        """Assess emergency risk"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            As an emergency medicine physician specializing in diabetes, assess:
            
            NUTRITION SUMMARY: {nutrition_summary}
            RISK SCORE: {risk_score}/10
            PATIENT PROFILE: {patient_profile}
            
            Determine:
            1. Emergency risk level (None, Low, Moderate, High, Critical)
            2. Warning signs to watch for
            3. Timeline for monitoring
            4. When to seek medical help
            5. Emergency action protocols
            6. Prevention strategies
            
            Be specific about:
            - Blood sugar thresholds
            - Symptom recognition
            - Contact protocols
            - Emergency supplies needed
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error assessing emergency risk: {str(e)}"

class DiabetesAdvisorAgent:
    """Agent responsible for providing diabetes-specific advice and management recommendations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Create specialized tools
        self.tools = [
            BloodSugarPredictionTool(),
            MealTimingTool(),
            PersonalizationTool(),
            EmergencyAssessmentTool()
        ]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diabetes management specialist providing comprehensive care advice.
            
            Your responsibilities:
            1. Predict blood sugar responses to meals
            2. Optimize meal timing for diabetes management
            3. Provide personalized advice based on patient profiles
            4. Assess emergency risks and provide protocols
            5. Generate actionable recommendations for daily management
            6. Coordinate care with healthcare providers
            
            Key principles:
            - Patient safety is paramount
            - Individualize all recommendations
            - Provide clear, actionable guidance
            - Consider lifestyle and preferences
            - Encourage self-advocacy and education
            - Know when to refer to healthcare providers
            
            Always consider the complete patient context when making recommendations."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def generate_diabetes_report(self, nutrition_summary: MealNutritionSummary, 
                               patient_profile: Optional[DiabetesProfile] = None) -> DiabetesAdvisoryReport:
        """Generate comprehensive diabetes management report"""
        try:
            # Use default profile if none provided
            if not patient_profile:
                patient_profile = self._get_default_profile()
            
            # Generate comprehensive analysis
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Create analysis input
            analysis_input = f"""
            Generate a comprehensive diabetes management report for this meal:
            
            NUTRITION SUMMARY:
            - Total calories: {nutrition_summary.total_calories}
            - Carbohydrates: {nutrition_summary.total_carbs_g}g
            - Sugars: {nutrition_summary.total_sugars_g}g
            - Fiber: {nutrition_summary.total_fiber_g}g
            - Protein: {nutrition_summary.total_protein_g}g
            - Fat: {nutrition_summary.total_fat_g}g
            - Glycemic Load: {nutrition_summary.total_glycemic_load}
            - Risk Score: {nutrition_summary.diabetes_risk_score}/10
            
            INDIVIDUAL FOODS:
            {self._format_food_list(nutrition_summary.individual_foods)}
            
            PATIENT PROFILE:
            {self._format_patient_profile(patient_profile)}
            
            CURRENT TIME: {current_time}
            
            Provide complete analysis including:
            1. Blood sugar response prediction
            2. Optimal meal timing
            3. Personalized recommendations
            4. Emergency risk assessment
            5. Daily tracking guidance
            6. Healthcare provider alerts
            """
            
            # Get comprehensive analysis from agent
            result = self.agent_executor.invoke({"input": analysis_input})
            
            # Parse and structure the result
            return self._parse_advisory_result(nutrition_summary, result["output"])
            
        except Exception as e:
            # Return error report
            return DiabetesAdvisoryReport(
                meal_summary=nutrition_summary,
                meal_timing=MealTiming(
                    optimal_timing="Analysis unavailable",
                    spacing_recommendation="Standard recommendations apply",
                    medication_timing=[],
                    exercise_timing="Standard recommendations apply"
                ),
                blood_sugar_prediction=BloodSugarPrediction(
                    predicted_peak_time="Unknown",
                    predicted_peak_range="Monitor closely",
                    duration_elevated="Unknown",
                    risk_level="Unknown",
                    monitoring_advice=[f"Analysis failed: {str(e)}"]
                ),
                personalized_advice=PersonalizedAdvice(
                    immediate_actions=["Consult healthcare provider"],
                    meal_modifications=[],
                    future_planning=[],
                    emergency_protocols=["Follow standard diabetes emergency protocols"],
                    lifestyle_tips=[]
                ),
                daily_tracking_notes=["System analysis unavailable"],
                healthcare_alerts=[f"System error: {str(e)}"]
            )
    
    def _get_default_profile(self) -> DiabetesProfile:
        """Get default patient profile when none provided"""
        return DiabetesProfile(
            diabetes_type="Type 2",
            current_medication=["Unknown"],
            target_blood_sugar={"fasting": 100, "post_meal": 140},
            activity_level="moderately_active",
            dietary_restrictions=[],
            last_hba1c=None
        )
    
    def _format_food_list(self, foods) -> str:
        """Format food list for analysis"""
        formatted = []
        for food in foods:
            formatted.append(f"- {food.food_name} ({food.serving_size}): "
                           f"{food.nutrition_per_serving.calories_per_100g * food.estimated_weight_g / 100:.0f} cal, "
                           f"{food.nutrition_per_serving.total_carbs_g * food.estimated_weight_g / 100:.1f}g carbs")
        return "\n".join(formatted)
    
    def _format_patient_profile(self, profile: DiabetesProfile) -> str:
        """Format patient profile for analysis"""
        return f"""
        - Diabetes Type: {profile.diabetes_type}
        - Medications: {', '.join(profile.current_medication)}
        - Activity Level: {profile.activity_level}
        - Target Blood Sugar: {profile.target_blood_sugar}
        - Dietary Restrictions: {', '.join(profile.dietary_restrictions) if profile.dietary_restrictions else 'None'}
        - Last HbA1c: {profile.last_hba1c if profile.last_hba1c else 'Unknown'}
        """
    
    def _parse_advisory_result(self, nutrition_summary: MealNutritionSummary, 
                             analysis_text: str) -> DiabetesAdvisoryReport:
        """Parse LLM analysis into structured advisory report"""
        # Simplified parsing - in production, use more sophisticated extraction
        
        return DiabetesAdvisoryReport(
            meal_summary=nutrition_summary,
            meal_timing=MealTiming(
                optimal_timing=self._extract_section(analysis_text, "timing", "Within 2 hours"),
                spacing_recommendation="3-4 hours between meals",
                medication_timing=["Take as prescribed by healthcare provider"],
                exercise_timing="Light walk 30 minutes after eating"
            ),
            blood_sugar_prediction=BloodSugarPrediction(
                predicted_peak_time=self._extract_section(analysis_text, "peak", "1-2 hours post-meal"),
                predicted_peak_range=self._extract_section(analysis_text, "range", "140-180 mg/dL"),
                duration_elevated="2-4 hours",
                risk_level=self._assess_risk_level(nutrition_summary.diabetes_risk_score),
                monitoring_advice=self._extract_monitoring_advice(analysis_text)
            ),
            personalized_advice=PersonalizedAdvice(
                immediate_actions=self._extract_actions(analysis_text, "immediate"),
                meal_modifications=self._extract_actions(analysis_text, "modification"),
                future_planning=self._extract_actions(analysis_text, "planning"),
                emergency_protocols=self._extract_actions(analysis_text, "emergency"),
                lifestyle_tips=self._extract_actions(analysis_text, "lifestyle")
            ),
            daily_tracking_notes=self._generate_tracking_notes(nutrition_summary),
            healthcare_alerts=self._generate_healthcare_alerts(nutrition_summary)
        )
    
    def _extract_section(self, text: str, keyword: str, default: str) -> str:
        """Extract specific section from analysis text"""
        lines = text.split('\n')
        for line in lines:
            if keyword.lower() in line.lower():
                return line.strip()
        return default
    
    def _assess_risk_level(self, risk_score: int) -> str:
        """Convert numeric risk score to text level"""
        if risk_score >= 8:
            return "High"
        elif risk_score >= 6:
            return "Moderate"
        elif risk_score >= 4:
            return "Low"
        else:
            return "Minimal"
    
    def _extract_monitoring_advice(self, text: str) -> List[str]:
        """Extract monitoring recommendations"""
        advice = []
        if "high" in text.lower():
            advice.append("Check blood sugar 1-2 hours after eating")
            advice.append("Monitor for symptoms of hyperglycemia")
        else:
            advice.append("Standard monitoring as per your routine")
        
        advice.append("Record meal details in diabetes log")
        return advice
    
    def _extract_actions(self, text: str, action_type: str) -> List[str]:
        """Extract specific action recommendations"""
        # Simplified extraction - in production, use more sophisticated NLP
        default_actions = {
            "immediate": ["Monitor blood sugar", "Stay hydrated"],
            "modification": ["Consider smaller portions", "Add more fiber"],
            "planning": ["Plan similar meals carefully", "Consider timing"],
            "emergency": ["Contact healthcare provider if blood sugar >250 mg/dL"],
            "lifestyle": ["Maintain regular meal schedule", "Stay active"]
        }
        
        return default_actions.get(action_type, ["Follow standard diabetes management protocols"])
    
    def _generate_tracking_notes(self, summary: MealNutritionSummary) -> List[str]:
        """Generate notes for daily tracking"""
        notes = []
        notes.append(f"Meal: {len(summary.individual_foods)} items, {summary.total_calories:.0f} calories")
        notes.append(f"Carbs: {summary.total_carbs_g:.1f}g, Sugars: {summary.total_sugars_g:.1f}g")
        
        if summary.total_glycemic_load:
            notes.append(f"Glycemic Load: {summary.total_glycemic_load:.1f}")
        
        notes.append(f"Risk Score: {summary.diabetes_risk_score}/10")
        
        return notes
    
    def _generate_healthcare_alerts(self, summary: MealNutritionSummary) -> List[str]:
        """Generate alerts for healthcare provider"""
        alerts = []
        
        if summary.diabetes_risk_score >= 8:
            alerts.append("HIGH RISK MEAL: Discuss meal planning strategies")
        
        if summary.total_sugars_g > 30:
            alerts.append(f"High sugar intake: {summary.total_sugars_g:.1f}g")
        
        if summary.total_carbs_g > 60:
            alerts.append(f"High carbohydrate meal: {summary.total_carbs_g:.1f}g")
        
        if not alerts:
            alerts.append("No immediate concerns noted")
        
        return alerts
