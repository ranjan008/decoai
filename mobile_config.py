"""
Mobile-optimized configuration for the Diabetes AI app
"""

# Mobile-friendly Streamlit configuration
MOBILE_CONFIG = {
    "page_title": "Diabetes AI Assistant",
    "page_icon": "🩺",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",  # Better for mobile
    "menu_items": {
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Diabetes AI Assistant\nAI-powered food analysis for diabetes management"
    }
}

# Mobile camera settings
CAMERA_CONFIG = {
    "max_image_size": 2 * 1024 * 1024,  # 2MB for mobile
    "image_quality": 0.8,
    "supported_formats": ["jpg", "jpeg", "png"],
    "camera_resolution": (1920, 1080)
}

# Mobile UI adjustments
MOBILE_UI = {
    "compact_mode": True,
    "touch_friendly": True,
    "swipe_navigation": True,
    "reduced_animations": True
}
