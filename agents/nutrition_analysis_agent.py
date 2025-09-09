import json
import requests
from typing import Dict, List, Optional, Tuple
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import config
from agents.food_recognition_agent import FoodItem

class NutritionInfo(BaseModel):
    """Nutritional information for a food item"""
    calories_per_100g: float = Field(description="Calories per 100 grams")
    total_carbs_g: float = Field(description="Total carbohydrates in grams")
    sugars_g: float = Field(description="Total sugars in grams")
    fiber_g: float = Field(description="Dietary fiber in grams")
    protein_g: float = Field(description="Protein content in grams")
    fat_g: float = Field(description="Total fat content in grams")
    sodium_mg: float = Field(description="Sodium content in milligrams")
    glycemic_index: Optional[int] = Field(description="Glycemic index (0-100)")
    glycemic_load: Optional[float] = Field(description="Glycemic load")

class FoodNutrition(BaseModel):
    """Complete nutrition data for a specific food item and quantity"""
    food_name: str = Field(description="Name of the food item")
    serving_size: str = Field(description="Actual serving size analyzed")
    estimated_weight_g: float = Field(description="Estimated weight in grams")
    nutrition_per_serving: NutritionInfo = Field(description="Nutrition for this specific serving")
    diabetes_impact: str = Field(description="Impact assessment for diabetes management")

class MealNutritionSummary(BaseModel):
    """Complete nutritional summary for an entire meal"""
    individual_foods: List[FoodNutrition] = Field(description="Nutrition for each food item")
    total_calories: float = Field(description="Total calories in the meal")
    total_carbs_g: float = Field(description="Total carbohydrates")
    total_sugars_g: float = Field(description="Total sugars")
    total_fiber_g: float = Field(description="Total fiber")
    total_protein_g: float = Field(description="Total protein")
    total_fat_g: float = Field(description="Total fat")
    average_glycemic_index: Optional[float] = Field(description="Weighted average GI")
    total_glycemic_load: Optional[float] = Field(description="Total glycemic load")
    diabetes_risk_score: int = Field(description="Risk score for diabetics (1-10)")
    meal_recommendations: List[str] = Field(description="Specific recommendations")

class NutritionDatabaseTool(BaseTool):
    """Tool for looking up nutritional information"""
    name: str = "lookup_nutrition"
    description: str = "Look up detailed nutritional information for food items"
    
    def _run(self, food_name: str, quantity: str = "100g") -> str:
        """Look up nutrition data for a food item"""
        try:
            # Use GPT-4 as a nutrition database (in production, you'd use a real API)
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            As a certified nutritionist, provide detailed nutritional information for:
            Food: {food_name}
            Quantity: {quantity}
            
            Include:
            1. Calories per 100g
            2. Macronutrients (carbs, protein, fat) in grams per 100g
            3. Sugar content in grams per 100g
            4. Fiber content in grams per 100g
            5. Sodium content in mg per 100g
            6. Glycemic Index (if known)
            7. Glycemic Load calculation
            8. Any diabetes-specific concerns
            
            Be precise and use standard USDA nutrition data when possible.
            If exact data isn't available, provide best estimates based on similar foods.
            
            Format as structured data with clear values.
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error looking up nutrition data: {str(e)}"

class PortionEstimationTool(BaseTool):
    """Tool for estimating portion sizes and weights"""
    name: str = "estimate_portion_weight"
    description: str = "Estimate the weight in grams of food portions based on visual descriptions"
    
    def _run(self, food_name: str, portion_description: str) -> str:
        """Estimate portion weight from description"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            As a portion size expert, estimate the weight in grams for:
            Food: {food_name}
            Portion Description: {portion_description}
            
            Consider:
            - Standard serving sizes for this food type
            - Density and typical preparation methods
            - Common household measurements
            - Visual cues from the description
            
            Provide:
            1. Estimated weight in grams
            2. Confidence level (1-10)
            3. Reasoning for the estimate
            4. Alternative measurements (cups, pieces, etc.)
            
            Be conservative for diabetes management - better to slightly overestimate than underestimate.
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error estimating portion: {str(e)}"

class GlycemicImpactTool(BaseTool):
    """Tool for calculating glycemic impact"""
    name: str = "calculate_glycemic_impact"
    description: str = "Calculate glycemic index and load for foods"
    
    def _run(self, food_name: str, carb_content_g: float, portion_weight_g: float) -> str:
        """Calculate glycemic impact"""
        try:
            client = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
            
            prompt = f"""
            Calculate the glycemic impact for:
            Food: {food_name}
            Carbohydrate content: {carb_content_g}g per 100g
            Portion weight: {portion_weight_g}g
            
            Provide:
            1. Glycemic Index (GI) value
            2. Glycemic Load (GL) calculation: (GI × carbs per serving) ÷ 100
            3. Classification (Low: GL<10, Medium: GL 10-19, High: GL≥20)
            4. Blood sugar impact explanation
            5. Timing recommendations for diabetics
            
            Use established GI values from scientific literature.
            """
            
            response = client.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            return f"Error calculating glycemic impact: {str(e)}"

class NutritionAnalysisAgent:
    """Agent responsible for analyzing nutritional content of identified foods"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Create specialized tools
        self.tools = [
            NutritionDatabaseTool(),
            PortionEstimationTool(),
            GlycemicImpactTool()
        ]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a nutrition analysis specialist focused on diabetes management.
            
            Your responsibilities:
            1. Calculate accurate nutritional values for identified foods
            2. Estimate portion weights from visual descriptions
            3. Determine glycemic index and load values
            4. Assess diabetes impact for each food item
            5. Provide total meal nutritional summary
            6. Generate specific recommendations for diabetic patients
            
            Key priorities:
            - Accuracy in carbohydrate and sugar calculations
            - Proper glycemic impact assessment
            - Conservative portion estimates for safety
            - Clear diabetes risk communication
            
            Use all available tools to provide comprehensive analysis."""),
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
    
    def analyze_meal_nutrition(self, foods: List[FoodItem]) -> MealNutritionSummary:
        """Analyze nutritional content for a complete meal"""
        try:
            # Analyze each food item individually
            food_analyses = []
            
            for food in foods:
                food_nutrition = self._analyze_single_food(food)
                food_analyses.append(food_nutrition)
            
            # Calculate meal totals
            meal_summary = self._calculate_meal_totals(food_analyses)
            
            return meal_summary
            
        except Exception as e:
            # Return error result
            return MealNutritionSummary(
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
                meal_recommendations=[f"Analysis failed: {str(e)}"]
            )
    
    def _analyze_single_food(self, food: FoodItem) -> FoodNutrition:
        """Analyze nutrition for a single food item"""
        try:
            # Get nutrition data using agent
            analysis_input = f"""
            Analyze the nutritional content for:
            Food: {food.name}
            Portion: {food.quantity}
            Description: {food.description}
            Confidence: {food.confidence}
            
            Provide complete nutritional analysis including:
            1. Estimated portion weight
            2. Complete macro and micronutrient breakdown
            3. Glycemic index and load
            4. Diabetes impact assessment
            """
            
            result = self.agent_executor.invoke({"input": analysis_input})
            
            # Parse the result into structured data
            return self._parse_nutrition_result(food, result["output"])
            
        except Exception as e:
            # Return default/error nutrition data
            return FoodNutrition(
                food_name=food.name,
                serving_size=food.quantity,
                estimated_weight_g=100.0,  # Default weight
                nutrition_per_serving=NutritionInfo(
                    calories_per_100g=200,
                    total_carbs_g=20,
                    sugars_g=10,
                    fiber_g=3,
                    protein_g=10,
                    fat_g=8,
                    sodium_mg=300,
                    glycemic_index=55,
                    glycemic_load=11
                ),
                diabetes_impact=f"Analysis unavailable: {str(e)}"
            )
    
    def _parse_nutrition_result(self, food: FoodItem, analysis_text: str) -> FoodNutrition:
        """Parse LLM analysis into structured nutrition data"""
        # Simplified parsing - in production, use more sophisticated extraction
        
        return FoodNutrition(
            food_name=food.name,
            serving_size=food.quantity,
            estimated_weight_g=self._extract_number(analysis_text, ["weight", "grams"], 100),
            nutrition_per_serving=NutritionInfo(
                calories_per_100g=self._extract_number(analysis_text, ["calories"], 200),
                total_carbs_g=self._extract_number(analysis_text, ["carb", "carbohydrate"], 20),
                sugars_g=self._extract_number(analysis_text, ["sugar"], 10),
                fiber_g=self._extract_number(analysis_text, ["fiber"], 3),
                protein_g=self._extract_number(analysis_text, ["protein"], 10),
                fat_g=self._extract_number(analysis_text, ["fat"], 8),
                sodium_mg=self._extract_number(analysis_text, ["sodium"], 300),
                glycemic_index=int(self._extract_number(analysis_text, ["glycemic index", "GI"], 55)),
                glycemic_load=self._extract_number(analysis_text, ["glycemic load", "GL"], 11)
            ),
            diabetes_impact=self._extract_diabetes_impact(analysis_text)
        )
    
    def _extract_number(self, text: str, keywords: List[str], default: float) -> float:
        """Extract numerical values from analysis text"""
        try:
            import re
            for keyword in keywords:
                pattern = rf"{keyword}[:\s]*(\d+\.?\d*)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return default
        except:
            return default
    
    def _extract_diabetes_impact(self, text: str) -> str:
        """Extract diabetes impact assessment from text"""
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['diabetes', 'blood sugar', 'glucose', 'impact']):
                return line.strip()
        return "Standard diabetes considerations apply"
    
    def _calculate_meal_totals(self, food_analyses: List[FoodNutrition]) -> MealNutritionSummary:
        """Calculate total nutritional values for the meal"""
        if not food_analyses:
            return MealNutritionSummary(
                individual_foods=[],
                total_calories=0,
                total_carbs_g=0,
                total_sugars_g=0,
                total_fiber_g=0,
                total_protein_g=0,
                total_fat_g=0,
                average_glycemic_index=None,
                total_glycemic_load=None,
                diabetes_risk_score=1,
                meal_recommendations=["No food items to analyze"]
            )
        
        # Calculate totals
        total_calories = sum(f.nutrition_per_serving.calories_per_100g * f.estimated_weight_g / 100 for f in food_analyses)
        total_carbs = sum(f.nutrition_per_serving.total_carbs_g * f.estimated_weight_g / 100 for f in food_analyses)
        total_sugars = sum(f.nutrition_per_serving.sugars_g * f.estimated_weight_g / 100 for f in food_analyses)
        total_fiber = sum(f.nutrition_per_serving.fiber_g * f.estimated_weight_g / 100 for f in food_analyses)
        total_protein = sum(f.nutrition_per_serving.protein_g * f.estimated_weight_g / 100 for f in food_analyses)
        total_fat = sum(f.nutrition_per_serving.fat_g * f.estimated_weight_g / 100 for f in food_analyses)
        
        # Calculate weighted average GI and total GL
        gi_values = [f.nutrition_per_serving.glycemic_index for f in food_analyses if f.nutrition_per_serving.glycemic_index]
        gl_values = [f.nutrition_per_serving.glycemic_load for f in food_analyses if f.nutrition_per_serving.glycemic_load]
        
        avg_gi = sum(gi_values) / len(gi_values) if gi_values else None
        total_gl = sum(gl_values) if gl_values else None
        
        # Calculate diabetes risk score
        risk_score = self._calculate_diabetes_risk(total_carbs, total_sugars, total_gl)
        
        # Generate recommendations
        recommendations = self._generate_meal_recommendations(
            total_calories, total_carbs, total_sugars, total_gl, risk_score
        )
        
        return MealNutritionSummary(
            individual_foods=food_analyses,
            total_calories=round(total_calories, 1),
            total_carbs_g=round(total_carbs, 1),
            total_sugars_g=round(total_sugars, 1),
            total_fiber_g=round(total_fiber, 1),
            total_protein_g=round(total_protein, 1),
            total_fat_g=round(total_fat, 1),
            average_glycemic_index=round(avg_gi, 1) if avg_gi else None,
            total_glycemic_load=round(total_gl, 1) if total_gl else None,
            diabetes_risk_score=risk_score,
            meal_recommendations=recommendations
        )
    
    def _calculate_diabetes_risk(self, carbs: float, sugars: float, glycemic_load: Optional[float]) -> int:
        """Calculate diabetes risk score (1-10)"""
        risk = 1
        
        # High carb content
        if carbs > 60:
            risk += 3
        elif carbs > 30:
            risk += 2
        elif carbs > 15:
            risk += 1
        
        # High sugar content
        if sugars > 25:
            risk += 3
        elif sugars > 15:
            risk += 2
        elif sugars > 10:
            risk += 1
        
        # High glycemic load
        if glycemic_load:
            if glycemic_load > 20:
                risk += 2
            elif glycemic_load > 10:
                risk += 1
        
        return min(risk, 10)
    
    def _generate_meal_recommendations(self, calories: float, carbs: float, 
                                     sugars: float, gl: Optional[float], risk: int) -> List[str]:
        """Generate diabetes-specific meal recommendations"""
        recommendations = []
        
        if risk >= 8:
            recommendations.append("⚠️ HIGH RISK: This meal may cause significant blood sugar spikes")
            recommendations.append("Consider reducing portion sizes or splitting the meal")
        elif risk >= 6:
            recommendations.append("⚠️ MODERATE RISK: Monitor blood sugar closely after this meal")
        
        if sugars > config.DAILY_SUGAR_LIMIT:
            recommendations.append(f"Sugar content ({sugars:.1f}g) exceeds daily recommendation ({config.DAILY_SUGAR_LIMIT}g)")
        
        if carbs > 45:
            recommendations.append("High carbohydrate content - consider pairing with protein or healthy fats")
        
        if gl and gl > 20:
            recommendations.append("High glycemic load - eat slowly and consider pre-meal exercise")
        
        # Positive recommendations
        recommendations.append("💡 Drink water before and during the meal")
        recommendations.append("💡 Consider a 10-15 minute walk after eating")
        
        if not recommendations:
            recommendations.append("✅ This meal appears suitable for diabetes management")
        
        return recommendations
