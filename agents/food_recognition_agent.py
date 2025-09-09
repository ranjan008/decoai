import base64
import io
import os
from typing import Dict, List, Optional
from PIL import Image
import requests
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field
import config

class FoodItem(BaseModel):
    """Represents a food item identified in an image"""
    name: str = Field(description="Name of the food item")
    quantity: str = Field(description="Estimated quantity/portion size")
    confidence: float = Field(description="Confidence score (0-1)")
    description: str = Field(description="Brief description of the food item")

class FoodRecognitionResult(BaseModel):
    """Result of food recognition analysis"""
    foods: List[FoodItem] = Field(description="List of identified food items")
    meal_type: str = Field(description="Type of meal (breakfast, lunch, dinner, snack)")
    total_items: int = Field(description="Total number of food items identified")
    analysis_notes: str = Field(description="Additional observations about the meal")

class ImageAnalysisTool(BaseTool):
    """Tool for analyzing food images using GPT-4 Vision"""
    name: str = "analyze_food_image"
    description: str = "Analyze an image to identify food items, quantities, and meal characteristics"
    
    def _run(self, image_path: str) -> str:
        """Analyze food image and return structured results"""
        try:
            import openai
            
            # Determine image format and convert to base64
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                
                # Save to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=85)
                img_bytes.seek(0)
                
                # Encode to base64
                base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            # Create OpenAI client directly
            client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            
            # Create detailed prompt for food recognition
            prompt = """
            You are a nutrition expert analyzing a food image for diabetes management. 
            
            Please identify the food items in this image and provide ONLY a clear, structured list.
            
            Format your response exactly like this:
            
            FOOD ITEMS:
            - **Rice** (1 cup)
            - **Chicken curry** (1 serving)
            - **Vegetables** (1/2 cup)
            
            MEAL TYPE: lunch
            
            NOTES: Brief observation about the meal
            
            Be specific about actual food items visible. Only list real food items, not analysis categories.
            """
            
            # Send request to GPT-4 Vision
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

class FoodRecognitionAgent:
    """Agent responsible for recognizing food items in images"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Create tools
        self.tools = [ImageAnalysisTool()]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a food recognition specialist helping diabetic patients track their meals.
            
            Your responsibilities:
            1. Accurately identify all food items in images
            2. Estimate portion sizes using common measurements
            3. Classify meal types and cooking methods
            4. Note any diabetes-relevant details (high sugar, high carb items)
            5. Provide confidence scores for identifications
            
            Always be thorough and err on the side of caution for diabetes management.
            Use the image analysis tool to examine food photos in detail."""),
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
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
    
    def analyze_food_image(self, image_path: str) -> FoodRecognitionResult:
        """Analyze a food image and return structured results"""
        try:
            # Validate image
            if not self._validate_image(image_path):
                raise ValueError("Invalid image file")
            
            # Use the image analysis tool directly
            image_tool = ImageAnalysisTool()
            analysis_result = image_tool._run(image_path)
            
            # Parse and structure the result
            return self._parse_analysis_result(analysis_result)
            
        except Exception as e:
            # Return error result
            return FoodRecognitionResult(
                foods=[],
                meal_type="unknown",
                total_items=0,
                analysis_notes=f"Analysis failed: {str(e)}"
            )
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate image file format and size"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False
                
            with Image.open(image_path) as img:
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size > config.MAX_IMAGE_SIZE:
                    return False
                
                # Check format - be more lenient with format checking
                if img.format and img.format.upper() in ['JPEG', 'JPG', 'PNG', 'WEBP']:
                    return True
                
                # Also check file extension as backup
                file_ext = os.path.splitext(image_path)[1].lower()
                if file_ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    return True
                
                return False
        except Exception as e:
            print(f"Image validation error: {str(e)}")
            return False
    
    def _parse_analysis_result(self, analysis_text: str) -> FoodRecognitionResult:
        """Parse the LLM analysis result into structured data"""
        import re
        
        lines = analysis_text.split('\n')
        foods = []
        meal_type = "lunch"  # Default
        analysis_notes = analysis_text
        
        # More precise patterns to extract actual food items
        food_patterns = [
            # Pattern for bullet points with bold food names: - **Pizza** (2 slices)
            r'[-*]\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)',
            # Pattern for numbered lists: 1. Rice (1 cup)
            r'\d+\.\s*([A-Za-z][a-z\s]+?)\s*\(([^)]+)\)',
            # Pattern for simple format: Food: quantity
            r'^([A-Za-z][a-z\s]+?):\s*([^,\n]+)',
            # Pattern for dash format: - Food item, quantity
            r'[-*]\s*([A-Za-z][a-z\s]+?),\s*([^,\n]+)'
        ]
        
        # Common food keywords to validate we're actually getting food items
        food_keywords = [
            'rice', 'bread', 'pasta', 'pizza', 'chicken', 'beef', 'fish', 'vegetables', 
            'salad', 'soup', 'sandwich', 'burger', 'egg', 'cheese', 'milk', 'yogurt',
            'apple', 'banana', 'orange', 'fruit', 'potato', 'tomato', 'carrot',
            'beans', 'meat', 'protein', 'cereal', 'oatmeal', 'noodles', 'curry'
        ]
        
        # Words to exclude (not food items)
        exclude_words = [
            'analysis', 'visible', 'estimated', 'type', 'cooking', 'dietary', 'confidence',
            'details', 'diabetes', 'carbohydrate', 'protein', 'sugary', 'hidden', 'sources'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or len(line) < 3:
                continue
                
            # Try different patterns to extract food items
            for pattern in food_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        food_name = match[0].strip()
                        quantity = match[1].strip()
                    else:
                        continue
                    
                    # Clean up food name
                    food_name = food_name.replace('**', '').strip()
                    food_name_lower = food_name.lower()
                    
                    # Validate this is actually a food item
                    is_valid_food = (
                        len(food_name) > 2 and 
                        len(food_name) < 50 and  # Not too long
                        not any(exclude in food_name_lower for exclude in exclude_words) and
                        (any(keyword in food_name_lower for keyword in food_keywords) or
                         re.match(r'^[a-zA-Z\s]+$', food_name))  # Only letters and spaces
                    )
                    
                    if is_valid_food:
                        foods.append(FoodItem(
                            name=food_name,
                            quantity=quantity,
                            confidence=0.9,
                            description=f"Identified from image analysis"
                        ))
                        break
        
        # Extract meal type
        meal_keywords = {
            'breakfast': ['breakfast', 'morning'],
            'lunch': ['lunch', 'midday', 'afternoon'],
            'dinner': ['dinner', 'evening', 'night'],
            'snack': ['snack', 'light']
        }
        
        for line in lines:
            line_lower = line.lower()
            for meal, keywords in meal_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    meal_type = meal
                    break
        
        # If no foods found, try a simpler approach
        if not foods:
            # Look for any capitalized words that might be food names
            text_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b', analysis_text)
            potential_foods = ['Baati', 'Dal', 'Rice', 'Bread', 'Curry', 'Chicken', 'Fish', 'Vegetables']
            
            for word in text_words:
                if word in potential_foods or any(food.lower() in word.lower() for food in potential_foods):
                    foods.append(FoodItem(
                        name=word,
                        quantity="1 serving",
                        confidence=0.7,
                        description=f"Extracted from text: {word}"
                    ))
        
        return FoodRecognitionResult(
            foods=foods if foods else [FoodItem(
                name="Unidentified food",
                quantity="unknown", 
                confidence=0.5,
                description="Could not clearly identify food items"
            )],
            meal_type=meal_type,
            total_items=len(foods) if foods else 0,
            analysis_notes=analysis_notes
        )
