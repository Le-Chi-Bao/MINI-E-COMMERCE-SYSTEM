from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional
import secrets
import os
from dotenv import load_dotenv

from app.database import get_db

load_dotenv()

# API Key authentication (simple version for demo)
API_KEYS = {
    os.getenv("API_KEY", "demo_key_12345"): "admin",
    "test_key_67890": "user"
}

class PredictorDependency:
    """Dependency for phone price predictor"""
    _instance = None
    
    def __init__(self):
        self.predictor = None
    
    def __call__(self):
        if self.predictor is None:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
                from predictor import PhonePricePredictor

                self.predictor = PhonePricePredictor()
                print("✅ Predictor initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize predictor: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Prediction service is temporarily unavailable"
                )
        return self.predictor

# # Create singleton instance
# get_predictor = PredictorDependency()

# # ... rest of the code unchanged ...
# class PredictorDependency:
#     def __call__(self):
#         if self.predictor is None:
#             try:
#                 # ✅ THÊM ĐƯỜNG DẪN
#                 import sys
#                 import os
#                 sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
                
#                 from predictor import PhonePricePredictor  # ✅ IMPORT TRONG FUNCTION
#                 self.predictor = PhonePricePredictor()
#                 print("✅ Predictor initialized successfully")
#             except Exception as e:
#                 print(f"❌ Failed to initialize predictor: {e}")
#                 # Xử lý lỗi...