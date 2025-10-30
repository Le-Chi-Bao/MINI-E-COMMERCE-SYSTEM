from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from feast import FeatureStore
import os
from typing import Dict, Any, List, Optional
from enum import Enum

# Khởi tạo FastAPI app
app = FastAPI(
    title="Phone Prediction API",
    description="API for predicting phone features using ML models with flexible service selection",
    version="1.0.0"
)

# 🆕 Enum cho các dịch vụ dự đoán
class PredictionService(str, Enum):
    RECOMMENDER = "recommender"
    VALUE_DETECTOR = "value_detector"
    CAMERA_PREDICTOR = "camera_predictor"
    ALL = "all"

# 🆕 Enum cho input methods
class InputMethod(str, Enum):
    FEATURE_STORE = "feature_store"
    MANUAL = "manual"

# Khởi tạo đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, "..", "my_phone_features")
models_path = os.path.join(current_dir, "..", "models")

# Khởi tạo predictor
class PhonePredictor:
    def __init__(self):
        try:
            # Load Feast store
            self.fs = FeatureStore(repo_path=repo_path)
            
            # Load models và scalers
            self.models = {
                'recommender': joblib.load(os.path.join(models_path, "model_recommender.pkl")),
                'value_detector': joblib.load(os.path.join(models_path, "model_value.pkl")),
                'camera_predictor': joblib.load(os.path.join(models_path, "model_camera.pkl"))
            }
            
            self.scalers = {
                'recommender': joblib.load(os.path.join(models_path, "scaler_recommender.pkl")),
                'value_detector': joblib.load(os.path.join(models_path, "scaler_value.pkl")), 
                'camera_predictor': joblib.load(os.path.join(models_path, "scaler_camera.pkl"))
            }
            
            # 🆕 Feature requirements cho từng service
            self.service_requirements = {
                'recommender': {
                    'features': [
                        'ScreenSize', 'PPI', 'total_resolution', 'camera_score', 'has_telephoto', 
                        'has_ultrawide', 'popularity_score', 'value_score', 'price_segment', 
                        'has_warranty', 'NumberOfReview'
                    ],
                    'feature_refs': [
                        "phone_display:ScreenSize", "phone_display:PPI", "phone_display:total_resolution",
                        "phone_camera:camera_score", "phone_camera:has_telephoto", "phone_camera:has_ultrawide",
                        "phone_ratings:popularity_score", "phone_value:value_score", "phone_value:price_segment",
                        "phone_product:has_warranty", "phone_product:NumberOfReview"
                    ],
                    'output': 'overall_score'
                },
                'value_detector': {
                    'features': [
                        'value_score', 'price_segment', 'overall_score', 'display_score', 
                        'camera_rating', 'PPI', 'ScreenSize', 'camera_score', 
                        'main_camera_mp', 'NumberOfReview'
                    ],
                    'feature_refs': [
                        "phone_value:value_score", "phone_value:price_segment", 
                        "phone_ratings:overall_score", "phone_ratings:display_score",
                        "phone_ratings:camera_rating", "phone_display:PPI", "phone_display:ScreenSize",
                        "phone_camera:camera_score", "phone_camera:main_camera_mp",
                        "phone_product:NumberOfReview"
                    ],
                    'output': ['is_premium', 'premium_probability']
                },
                'camera_predictor': {
                    'features': [
                        'main_camera_mp', 'num_cameras', 'has_telephoto', 'has_ultrawide', 'has_ois', 
                        'camera_feature_count', 'PPI', 'total_resolution', 'ScreenSize', 'value_score', 
                        'is_premium', 'NumberOfReview'
                    ],
                    'feature_refs': [
                        "phone_camera:main_camera_mp", "phone_camera:num_cameras", 
                        "phone_camera:has_telephoto", "phone_camera:has_ultrawide", 
                        "phone_camera:has_ois", "phone_camera:camera_feature_count",
                        "phone_display:PPI", "phone_display:total_resolution", "phone_display:ScreenSize",
                        "phone_value:value_score", "phone_value:is_premium", 
                        "phone_product:NumberOfReview"
                    ],
                    'output': 'camera_rating'
                }
            }
            
            print("✅ Predictor initialized successfully!")
            print(f"   Available services: {list(self.service_requirements.keys())}")
            
        except Exception as e:
            print(f"❌ Error initializing predictor: {e}")
            raise

    def get_required_features(self, services: List[str]) -> List[str]:
        """Lấy danh sách features cần thiết cho các services"""
        all_features = []
        for service in services:
            if service in self.service_requirements:
                all_features.extend(self.service_requirements[service]['feature_refs'])
        return list(set(all_features))  # Remove duplicates

    def predict_service(self, service: str, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """Dự đoán cho một service cụ thể"""
        if service not in self.service_requirements:
            raise ValueError(f"Service {service} not supported")
        
        requirements = self.service_requirements[service]
        predictions = {}
        
        try:
            if service == 'recommender':
                X = feature_data[requirements['features']]
                X_scaled = self.scalers['recommender'].transform(X)
                predictions['overall_score'] = round(self.models['recommender'].predict(X_scaled)[0], 1)
                
            elif service == 'value_detector':
                X = feature_data[requirements['features']]
                X_scaled = self.scalers['value_detector'].transform(X)
                predictions['is_premium'] = bool(self.models['value_detector'].predict(X_scaled)[0])
                predictions['premium_probability'] = round(self.models['value_detector'].predict_proba(X_scaled)[0][1], 3)
                
            elif service == 'camera_predictor':
                X = feature_data[requirements['features']]
                X_scaled = self.scalers['camera_predictor'].transform(X)
                predictions['camera_rating'] = round(self.models['camera_predictor'].predict(X_scaled)[0], 1)
                
        except Exception as e:
            raise ValueError(f"Prediction failed for {service}: {str(e)}")
        
        return predictions

    def predict_manual(self, services: List[str], manual_features: Dict[str, float]) -> Dict[str, Any]:
        """Dự đoán với features nhập thủ công"""
        # Tạo DataFrame từ manual features
        feature_data = pd.DataFrame([manual_features])
        
        # Kiểm tra xem có đủ features không
        all_required_features = []
        for service in services:
            if service in self.service_requirements:
                all_required_features.extend(self.service_requirements[service]['features'])
        
        missing_features = set(all_required_features) - set(manual_features.keys())
        if missing_features:
            raise ValueError(f"Missing features for selected services: {missing_features}")
        
        # Thực hiện dự đoán
        results = {}
        for service in services:
            results.update(self.predict_service(service, feature_data))
        
        return results

    def predict_feature_store(self, services: List[str], product_id: str) -> Dict[str, Any]:
        """Dự đoán với features từ Feature Store"""
        # Lấy features cần thiết
        required_features = self.get_required_features(services)
        
        feature_data = self.fs.get_online_features(
            entity_rows=[{"product_id": product_id}],
            features=required_features
        ).to_df()

        # Thực hiện dự đoán
        results = {}
        for service in services:
            results.update(self.predict_service(service, feature_data))
        
        return results

# Khởi tạo predictor
predictor = PhonePredictor()

# 🆕 Pydantic models
class ManualFeatures(BaseModel):
    # Features cho Recommender
    ScreenSize: Optional[float] = None
    PPI: Optional[float] = None
    total_resolution: Optional[float] = None
    camera_score: Optional[float] = None
    has_telephoto: Optional[int] = None
    has_ultrawide: Optional[int] = None
    popularity_score: Optional[float] = None
    value_score: Optional[float] = None
    price_segment: Optional[int] = None
    has_warranty: Optional[int] = None
    NumberOfReview: Optional[int] = None
    
    # Features cho Value Detector
    overall_score: Optional[float] = None
    display_score: Optional[float] = None
    camera_rating: Optional[float] = None
    main_camera_mp: Optional[float] = None
    
    # Features cho Camera Predictor
    num_cameras: Optional[int] = None
    has_ois: Optional[int] = None
    camera_feature_count: Optional[int] = None
    is_premium: Optional[int] = None

class PredictionRequest(BaseModel):
    services: List[PredictionService]
    input_method: InputMethod
    product_id: Optional[str] = None
    manual_features: Optional[ManualFeatures] = None

class PredictionResponse(BaseModel):
    services: List[str]
    predictions: Dict[str, Any]
    input_method: str
    status: str

# Routes
@app.get("/")
async def root():
    return {
        "message": "Flexible Phone Prediction API", 
        "version": "1.0.0",
        "available_services": [service.value for service in PredictionService],
        "endpoints": {
            "health_check": "/health",
            "services_info": "/services",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": len(predictor.models),
        "available_services": list(predictor.service_requirements.keys())
    }

@app.get("/services")
async def get_services_info():
    """Lấy thông tin về các dịch vụ dự đoán available"""
    services_info = {}
    for service, requirements in predictor.service_requirements.items():
        services_info[service] = {
            "required_features": requirements['features'],
            "output": requirements['output'],
            "feature_count": len(requirements['features'])
        }
    
    return {
        "services": services_info,
        "total_services": len(services_info)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_phone(request: PredictionRequest):
    """Dự đoán linh hoạt với các service được chọn"""
    print(f"📥 Prediction request - Services: {request.services}, Method: {request.input_method}")
    
    try:
        # Chuyển services thành list
        service_list = [service.value for service in request.services]
        if 'all' in service_list:
            service_list = ['recommender', 'value_detector', 'camera_predictor']
        
        # Kiểm tra input method
        if request.input_method == InputMethod.FEATURE_STORE:
            if not request.product_id:
                raise HTTPException(status_code=400, detail="product_id is required for feature_store input method")
            
            predictions = predictor.predict_feature_store(service_list, request.product_id)
            input_info = f"feature_store:{request.product_id}"
            
        elif request.input_method == InputMethod.MANUAL:
            if not request.manual_features:
                raise HTTPException(status_code=400, detail="manual_features is required for manual input method")
            
            # Convert manual features to dict
            manual_features_dict = request.manual_features.dict(exclude_none=True)
            predictions = predictor.predict_manual(service_list, manual_features_dict)
            input_info = "manual_input"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid input method")
        
        return {
            'services': service_list,
            'predictions': predictions,
            'input_method': input_info,
            'status': 'success'
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 🆕 Endpoint đơn giản cho từng service
@app.get("/predict/{service}/{product_id}")
async def predict_single_service(service: PredictionService, product_id: str):
    """Dự đoán nhanh cho một service cụ thể"""
    service_list = [service.value]
    if service.value == 'all':
        service_list = ['recommender', 'value_detector', 'camera_predictor']
    
    try:
        predictions = predictor.predict_feature_store(service_list, product_id)
        
        return {
            'service': service.value,
            'product_id': product_id,
            'predictions': predictions,
            'status': 'success'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Flexible Phone Prediction API...")
    print(f"📁 Available services: {list(predictor.service_requirements.keys())}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )