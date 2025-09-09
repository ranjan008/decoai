# 🩺 Diabetes AI Assistant

An advanced agentic AI system built with LangChain that helps diabetic patients track their food intake through image analysis, providing detailed nutritional information and personalized diabetes management advice.

## ✨ Features

### 🔍 **Food Recognition**
- Advanced computer vision using GPT-4 Vision
- Identifies multiple food items in a single image
- Estimates portion sizes and cooking methods
- Confidence scoring for each identification

### 📊 **Nutrition Analysis**
- Detailed macro and micronutrient breakdown
- Calorie counting with diabetes-specific focus
- Glycemic Index and Glycemic Load calculation
- Sugar content analysis and daily tracking

### 🩺 **Diabetes Management**
- Personalized blood sugar impact predictions
- Meal timing optimization recommendations
- Risk assessment (1-10 scale)
- Emergency protocol guidance
- Healthcare provider alerts

### 🤖 **Agentic AI Architecture**
- **Food Recognition Agent**: Specialized in visual food identification
- **Nutrition Analysis Agent**: Expert in nutritional calculations
- **Diabetes Advisor Agent**: Focused on diabetes-specific recommendations
- **Orchestrator System**: Coordinates all agents using LangChain

## 🏗️ System Architecture

```
📸 Image Input
    ↓
🔍 Food Recognition Agent (GPT-4 Vision)
    ↓
📊 Nutrition Analysis Agent (GPT-4 + Nutrition Database)
    ↓
🩺 Diabetes Advisor Agent (GPT-4 + Medical Knowledge)
    ↓
📋 Comprehensive Report
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/api-keys)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd dicoai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Windows
set OPENAI_API_KEY=your_openai_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_openai_api_key_here
```

4. **Test the system:**
```bash
python test_system.py
```

5. **Run the web application:**
```bash
streamlit run app.py
```

## 📱 Usage

### Web Interface

1. **Launch the app:**
```bash
streamlit run app.py
```

2. **Set up patient profile** (optional but recommended):
   - Diabetes type (Type 1, Type 2, Gestational)
   - Current medications
   - Activity level
   - Target blood sugar ranges
   - Dietary restrictions

3. **Upload food image:**
   - Take a clear photo of your meal
   - Upload supported formats: JPG, PNG, WEBP
   - Add optional notes about the meal

4. **Get comprehensive analysis:**
   - Food identification results
   - Complete nutritional breakdown
   - Diabetes-specific recommendations
   - Blood sugar impact predictions

### Programmatic Usage

```python
from diabetes_ai_system import analyze_food_image

# Simple analysis
report = analyze_food_image("path/to/food/image.jpg")

# With patient profile
patient_profile = {
    "diabetes_type": "Type 2",
    "current_medication": ["Metformin"],
    "activity_level": "moderately_active",
    "target_blood_sugar": {"fasting": 100, "post_meal": 140}
}

report = analyze_food_image("path/to/food/image.jpg", patient_profile)

print(f"Total calories: {report.nutrition_analysis.total_calories}")
print(f"Risk score: {report.nutrition_analysis.diabetes_risk_score}/10")
```

## 📊 Report Components

### Food Recognition Results
- List of identified food items
- Portion size estimates
- Confidence scores
- Meal type classification

### Nutrition Analysis
- **Macronutrients**: Carbs, protein, fat, fiber
- **Micronutrients**: Sodium, vitamins (when available)
- **Diabetes Metrics**: Glycemic index, glycemic load
- **Risk Assessment**: 1-10 scale for diabetes impact

### Diabetes Advisory
- **Blood Sugar Prediction**: Expected peak times and ranges
- **Meal Timing**: Optimal eating schedules
- **Personalized Advice**: Based on patient profile
- **Emergency Protocols**: When to seek medical help
- **Monitoring Recommendations**: Testing schedules

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Required
FOOD_API_KEY=your_food_api_key_here      # Optional
```

### Customizable Settings (config.py)
```python
# Image processing
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]

# Diabetes thresholds
DAILY_SUGAR_LIMIT = 25  # grams
DAILY_CALORIE_TARGETS = {
    "sedentary_male": 2000,
    "active_female": 2000
}
```

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
# Run all tests
python test_system.py

# Check components only
python test_system.py --check-only
```

### Test Coverage
- ✅ System component verification
- ✅ Individual agent functionality
- ✅ End-to-end analysis pipeline
- ✅ Error handling and edge cases

## 📁 Project Structure

```
dicoai/
├── agents/                     # AI agents
│   ├── __init__.py
│   ├── food_recognition_agent.py
│   ├── nutrition_analysis_agent.py
│   └── diabetes_advisor_agent.py
├── diabetes_ai_system.py       # Main orchestrator
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration settings
├── test_system.py             # Test suite
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## 🔐 Security & Privacy

- **Local Processing**: Images are processed locally and not stored
- **API Security**: Secure communication with OpenAI
- **Data Privacy**: No personal health data is permanently stored
- **Temporary Files**: Automatically cleaned up after processing

## ⚠️ Medical Disclaimer

**IMPORTANT**: This AI assistant is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for:

- Diabetes management decisions
- Medication adjustments
- Emergency medical situations
- Treatment plan modifications

## 🛠️ Advanced Usage

### Custom Patient Profiles

```python
from agents.diabetes_advisor_agent import DiabetesProfile

profile = DiabetesProfile(
    diabetes_type="Type 1",
    current_medication=["Insulin", "Metformin"],
    target_blood_sugar={"fasting": 90, "post_meal": 130},
    activity_level="very_active",
    dietary_restrictions=["Gluten-free", "Low sodium"],
    last_hba1c=6.8
)
```

### Batch Processing

```python
from diabetes_ai_system import DiabetesAISystem

system = DiabetesAISystem()
image_paths = ["meal1.jpg", "meal2.jpg", "meal3.jpg"]

for path in image_paths:
    report = system.analyze_meal(MealAnalysisRequest(image_path=path))
    print(f"Analyzed {path}: {report.summary_score}/10")
```

## 🔄 API Integration

The system can be extended with external nutrition databases:

```python
# Example: USDA FoodData Central integration
# Add your preferred nutrition API key to config.py
FOOD_API_KEY = "your_usda_api_key"
```

## 📈 Performance Optimization

- **Caching**: Agent responses cached for similar food items
- **Parallel Processing**: Multiple food items analyzed simultaneously
- **Image Optimization**: Automatic resizing for faster processing
- **Error Recovery**: Graceful handling of API failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with healthcare regulations in your jurisdiction.

## 🆘 Support

For issues or questions:
1. Check the test suite output
2. Verify API key configuration
3. Review error logs in the console
4. Ensure all dependencies are installed correctly

---

**Built with ❤️ for diabetes management and healthcare innovation**

