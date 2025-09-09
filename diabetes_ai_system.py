"""
Diabetes AI System - Main orchestrator for agentic AI diabetes management
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field

import config
from agents.food_recognition_agent import FoodRecognitionAgent, FoodRecognitionResult
from agents.nutrition_analysis_agent import NutritionAnalysisAgent, MealNutritionSummary
from agents.diabetes_advisor_agent import DiabetesAdvisorAgent, DiabetesAdvisoryReport, DiabetesProfile

class MealAnalysisRequest(BaseModel):
    """Request for meal analysis"""
    image_path: str = Field(description="Path to the food image")
    meal_time: Optional[str] = Field(default=None, description="Time of meal (optional)")
    patient_notes: Optional[str] = Field(default=None, description="Any additional patient notes")
    patient_profile: Optional[DiabetesProfile] = Field(default=None, description="Patient diabetes profile")

class ComprehensiveReport(BaseModel):
    """Complete diabetes meal analysis report"""
    timestamp: str = Field(description="Analysis timestamp")
    food_recognition: FoodRecognitionResult = Field(description="Food identification results")
    nutrition_analysis: MealNutritionSummary = Field(description="Nutritional breakdown")
    diabetes_advisory: DiabetesAdvisoryReport = Field(description="Diabetes management advice")
    summary_score: int = Field(description="Overall meal score (1-10)")
    key_insights: List[str] = Field(description="Top 3-5 key insights")
    action_items: List[str] = Field(description="Immediate action items")

class FoodRecognitionTool(BaseTool):
    """Tool that wraps the food recognition agent"""
    name: str = "recognize_food_in_image"
    description: str = "Analyze an image to identify food items and their characteristics"
    
    def __init__(self):
        super().__init__()
        
    @property
    def food_agent(self):
        if not hasattr(self, '_food_agent'):
            self._food_agent = FoodRecognitionAgent()
        return self._food_agent
    
    def _run(self, image_path: str) -> str:
        """Run food recognition on image"""
        try:
            result = self.food_agent.analyze_food_image(image_path)
            return json.dumps(result.model_dump(), indent=2)
        except Exception as e:
            return f"Error in food recognition: {str(e)}"

class NutritionAnalysisTool(BaseTool):
    """Tool that wraps the nutrition analysis agent"""
    name: str = "analyze_nutrition"
    description: str = "Analyze nutritional content of identified foods"
    
    def __init__(self):
        super().__init__()
        
    @property
    def nutrition_agent(self):
        if not hasattr(self, '_nutrition_agent'):
            self._nutrition_agent = NutritionAnalysisAgent()
        return self._nutrition_agent
    
    def _run(self, food_recognition_result: str) -> str:
        """Run nutrition analysis on identified foods"""
        try:
            # Handle both JSON and string inputs
            if isinstance(food_recognition_result, str):
                try:
                    food_data = json.loads(food_recognition_result)
                    food_result = FoodRecognitionResult(**food_data)
                except json.JSONDecodeError:
                    # If not JSON, treat as comma-separated food names
                    food_names = [name.strip() for name in food_recognition_result.split(',')]
                    from agents.food_recognition_agent import FoodItem
                    foods = []
                    for name in food_names:
                        foods.append(FoodItem(
                            name=name,
                            quantity="1 serving",
                            confidence=0.8,
                            description=f"Identified food: {name}"
                        ))
                    food_result = FoodRecognitionResult(
                        foods=foods,
                        meal_type="lunch",
                        total_items=len(foods),
                        analysis_notes="Foods identified from text input"
                    )
            else:
                food_result = food_recognition_result
            
            # Analyze nutrition
            nutrition_summary = self.nutrition_agent.analyze_meal_nutrition(food_result.foods)
            return json.dumps(nutrition_summary.model_dump(), indent=2)
        except Exception as e:
            return f"Error in nutrition analysis: {str(e)}"

class DiabetesAdvisoryTool(BaseTool):
    """Tool that wraps the diabetes advisor agent"""
    name: str = "generate_diabetes_advice"
    description: str = "Generate diabetes-specific advice and management recommendations"
    
    def __init__(self):
        super().__init__()
        
    @property
    def advisor_agent(self):
        if not hasattr(self, '_advisor_agent'):
            self._advisor_agent = DiabetesAdvisorAgent()
        return self._advisor_agent
    
    def _run(self, nutrition_summary: str, patient_profile: str = None) -> str:
        """Generate diabetes advisory report"""
        try:
            # Parse nutrition summary
            nutrition_data = json.loads(nutrition_summary)
            meal_summary = MealNutritionSummary(**nutrition_data)
            
            # Parse patient profile if provided
            profile = None
            if patient_profile:
                try:
                    profile_data = json.loads(patient_profile)
                    profile = DiabetesProfile(**profile_data)
                except:
                    pass  # Use default profile
            
            # Generate advisory report
            advisory_report = self.advisor_agent.generate_diabetes_report(meal_summary, profile)
            return json.dumps(advisory_report.model_dump(), indent=2)
        except Exception as e:
            return f"Error in diabetes advisory: {str(e)}"

class DiabetesAISystem:
    """Main system orchestrator for diabetes meal analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Initialize specialized agents
        self.food_recognition_agent = FoodRecognitionAgent()
        self.nutrition_agent = NutritionAnalysisAgent()
        self.advisor_agent = DiabetesAdvisorAgent()
        
        # Create orchestration tools
        self.tools = [
            FoodRecognitionTool(),
            NutritionAnalysisTool(),
            DiabetesAdvisoryTool()
        ]
        
        # Create main orchestrator prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized diabetes management AI assistant. You MUST analyze food images using your available tools.

            CRITICAL: You have access to these tools and MUST use them for every food image analysis:
            1. recognize_food_in_image - Use this to analyze the food image
            2. analyze_nutrition - Use this to calculate nutrition from recognized foods
            3. generate_diabetes_advice - Use this to create diabetes recommendations

            MANDATORY PROCESS for every food image request:
            Step 1: ALWAYS call recognize_food_in_image with the image path provided
            Step 2: ALWAYS call analyze_nutrition with the food recognition results
            Step 3: ALWAYS call generate_diabetes_advice with the nutrition analysis
            Step 4: Summarize the results

            NEVER provide generic responses about being unable to analyze images.
            NEVER skip the tool usage steps.
            ALWAYS use the tools to provide real food analysis and diabetes advice.

            You are fully capable of analyzing food images through these specialized tools."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create main orchestrator agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=10
        )
    
    def analyze_meal(self, request: MealAnalysisRequest) -> ComprehensiveReport:
        """
        Main entry point for meal analysis
        
        Args:
            request: MealAnalysisRequest containing image path and optional patient info
            
        Returns:
            ComprehensiveReport with complete analysis results
        """
        try:
            # Validate image exists
            if not os.path.exists(request.image_path):
                raise FileNotFoundError(f"Image not found: {request.image_path}")
            
            print(f"🔍 Starting meal analysis for: {request.image_path}")
            
            # Step 1: Direct food recognition (bypass agent executor)
            print("📸 Step 1: Analyzing food image...")
            food_tool = FoodRecognitionTool()
            food_result_json = food_tool._run(request.image_path)
            food_recognition_result = json.loads(food_result_json)
            print(f"✅ Found {len(food_recognition_result.get('food_items', []))} food items")
            
            # Step 2: Direct nutrition analysis
            print("🍎 Step 2: Analyzing nutrition...")
            nutrition_tool = NutritionAnalysisTool()
            nutrition_result_json = nutrition_tool._run(food_result_json)
            nutrition_summary = json.loads(nutrition_result_json)
            print(f"✅ Calculated nutrition: {nutrition_summary.get('total_calories', 0)} calories")
            
            # Step 3: Direct diabetes advisory
            print("🩺 Step 3: Generating diabetes advice...")
            advisory_tool = DiabetesAdvisoryTool()
            patient_profile_json = json.dumps(request.patient_profile.model_dump()) if request.patient_profile else "{}"
            advisory_result_json = advisory_tool._run(nutrition_result_json, patient_profile_json)
            advisory_report = json.loads(advisory_result_json)
            print("✅ Generated diabetes recommendations")
            
            # Create comprehensive report
            return self._create_comprehensive_report(
                food_recognition_result,
                nutrition_summary,
                advisory_report,
                "Analysis completed successfully using direct tool execution"
            )
            
        except Exception as e:
            # Return error report
            return ComprehensiveReport(
                timestamp=datetime.now().isoformat(),
                food_recognition=FoodRecognitionResult(
                    foods=[],
                    meal_type="unknown",
                    total_items=0,
                    analysis_notes=f"Analysis failed: {str(e)}"
                ),
                nutrition_analysis=MealNutritionSummary(
                    individual_foods=[],
                    total_calories=0,
                    total_carbs_g=0,
                    total_sugars_g=0,
                    total_fiber_g=0,
                    total_protein_g=0,
                    total_fat_g=0,
                    average_glycemic_index=None,
                    total_glycemic_load=None,
                    diabetes_risk_score=5,
                    meal_recommendations=[f"System error: {str(e)}"]
                ),
                diabetes_advisory=DiabetesAdvisoryReport(
                    meal_summary=MealNutritionSummary(
                        individual_foods=[], total_calories=0, total_carbs_g=0,
                        total_sugars_g=0, total_fiber_g=0, total_protein_g=0,
                        total_fat_g=0, average_glycemic_index=None,
                        total_glycemic_load=None, diabetes_risk_score=5,
                        meal_recommendations=[]
                    ),
                    daily_tracking_notes=[],
                    healthcare_alerts=[f"System error: {str(e)}"]
                ),
                summary_score=1,
                key_insights=[f"Analysis failed: {str(e)}"],
                action_items=["Please try again or contact support"]
            )
    
    def _prepare_analysis_input(self, request: MealAnalysisRequest) -> str:
        """Prepare input for the orchestrator agent"""
        input_parts = [
            f"Analyze the diabetes meal image at: {request.image_path}",
            f"Analysis timestamp: {datetime.now().isoformat()}"
        ]
        
        if request.meal_time:
            input_parts.append(f"Meal time: {request.meal_time}")
        
        if request.patient_notes:
            input_parts.append(f"Patient notes: {request.patient_notes}")
        
        if request.patient_profile:
            input_parts.append(f"Patient profile: {json.dumps(request.patient_profile.dict())}")
        
        input_parts.extend([
            "",
            "Please complete the full analysis pipeline:",
            "1. Recognize and identify all food items in the image",
            "2. Analyze the nutritional content and diabetes impact",
            "3. Generate personalized diabetes management recommendations",
            "4. Provide a comprehensive summary with key insights and action items"
        ])
        
        return "\n".join(input_parts)
    
    def _create_comprehensive_report(self, food_result: Dict, nutrition_result: Dict,
                                   advisory_result: Dict, summary_text: str) -> ComprehensiveReport:
        """Create final comprehensive report from all analysis results"""
        
        # Parse results into proper objects
        food_recognition = FoodRecognitionResult(**food_result) if food_result else None
        nutrition_analysis = MealNutritionSummary(**nutrition_result) if nutrition_result else None
        diabetes_advisory = DiabetesAdvisoryReport(**advisory_result) if advisory_result else None
        
        # Calculate summary score
        summary_score = self._calculate_summary_score(nutrition_analysis)
        
        # Extract key insights
        key_insights = self._extract_key_insights(food_recognition, nutrition_analysis, diabetes_advisory)
        
        # Generate action items
        action_items = self._generate_action_items(nutrition_analysis, diabetes_advisory)
        
        return ComprehensiveReport(
            timestamp=datetime.now().isoformat(),
            food_recognition=food_recognition or FoodRecognitionResult(
                foods=[], meal_type="unknown", total_items=0, analysis_notes="No data"
            ),
            nutrition_analysis=nutrition_analysis or MealNutritionSummary(
                individual_foods=[], total_calories=0, total_carbs_g=0,
                total_sugars_g=0, total_fiber_g=0, total_protein_g=0,
                total_fat_g=0, average_glycemic_index=None,
                total_glycemic_load=None, diabetes_risk_score=5,
                meal_recommendations=[]
            ),
            diabetes_advisory=diabetes_advisory or DiabetesAdvisoryReport(
                meal_summary=nutrition_analysis,
                daily_tracking_notes=[], healthcare_alerts=[]
            ),
            summary_score=summary_score,
            key_insights=key_insights,
            action_items=action_items
        )
    
    def _calculate_summary_score(self, nutrition: Optional[MealNutritionSummary]) -> int:
        """Calculate overall meal score for diabetes management (1-10)"""
        if not nutrition:
            return 5
        
        # Higher score = better for diabetes management
        score = 10 - nutrition.diabetes_risk_score
        return max(1, min(10, score))
    
    def _extract_key_insights(self, food_result: Optional[FoodRecognitionResult],
                            nutrition: Optional[MealNutritionSummary],
                            advisory: Optional[DiabetesAdvisoryReport]) -> List[str]:
        """Extract top key insights from analysis"""
        insights = []
        
        if nutrition:
            insights.append(f"📊 Total: {nutrition.total_calories:.0f} cal, {nutrition.total_carbs_g:.1f}g carbs, {nutrition.total_sugars_g:.1f}g sugars")
            
            if nutrition.diabetes_risk_score >= 7:
                insights.append("⚠️ HIGH diabetes risk - monitor blood sugar closely")
            elif nutrition.diabetes_risk_score >= 5:
                insights.append("⚠️ MODERATE diabetes risk - standard monitoring recommended")
            else:
                insights.append("✅ LOW diabetes risk - suitable for diabetes management")
            
            if nutrition.total_glycemic_load and nutrition.total_glycemic_load > 20:
                insights.append(f"📈 High glycemic load ({nutrition.total_glycemic_load:.1f}) - expect blood sugar spike")
        
        if food_result and food_result.foods:
            insights.append(f"🍽️ Identified {len(food_result.foods)} food items in {food_result.meal_type}")
        
        if advisory and advisory.personalized_advice:
            if advisory.personalized_advice.immediate_actions:
                insights.append(f"⚡ Priority: {advisory.personalized_advice.immediate_actions[0]}")
        
        return insights[:5]  # Return top 5 insights
    
    def _generate_action_items(self, nutrition: Optional[MealNutritionSummary],
                             advisory: Optional[DiabetesAdvisoryReport]) -> List[str]:
        """Generate immediate action items"""
        actions = []
        
        if nutrition and nutrition.diabetes_risk_score >= 7:
            actions.append("🔍 Check blood sugar 1-2 hours after eating")
            actions.append("💧 Drink plenty of water")
        
        if nutrition and nutrition.total_sugars_g > 25:
            actions.append("⏰ Consider light exercise after meal")
        
        if advisory and advisory.personalized_advice:
            actions.extend(advisory.personalized_advice.immediate_actions[:2])
        
        if not actions:
            actions.append("📝 Log meal details in diabetes tracking app")
            actions.append("🚶 Consider a 10-minute walk after eating")
        
        return actions[:5]  # Return top 5 actions
    
    def save_report(self, report: ComprehensiveReport, output_path: str) -> str:
        """Save comprehensive report to file"""
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.dict(), f, indent=2, ensure_ascii=False)
            
            return f"Report saved to: {output_path}"
        except Exception as e:
            return f"Error saving report: {str(e)}"

# Convenience function for quick analysis
def analyze_food_image(image_path: str, patient_profile: Optional[Dict] = None) -> ComprehensiveReport:
    """
    Quick analysis function for food images
    
    Args:
        image_path: Path to the food image
        patient_profile: Optional patient profile dictionary
        
    Returns:
        ComprehensiveReport with complete analysis
    """
    system = DiabetesAISystem()
    
    # Convert patient profile if provided
    profile = None
    if patient_profile:
        try:
            profile = DiabetesProfile(**patient_profile)
        except:
            pass  # Use default profile
    
    request = MealAnalysisRequest(
        image_path=image_path,
        patient_profile=profile
    )
    
    return system.analyze_meal(request)
