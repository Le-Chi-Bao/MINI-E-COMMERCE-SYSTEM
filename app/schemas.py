from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PhoneFeatures(BaseModel):
    screen_size: float = Field(..., ge=4.0, le=8.0)
    resolution_width: int = Field(..., ge=720, le=3840)
    resolution_height: int = Field(..., ge=1280, le=2160)
    main_camera_mp: float = Field(..., ge=5.0, le=200.0)
    num_cameras: int = Field(..., ge=1, le=5)
    has_telephoto: bool = False
    has_ultrawide: bool = False
    has_ois: bool = False
    has_warranty: bool = False
    number_of_reviews: float = Field(0.0, ge=0.0)

class PredictionRequest(BaseModel):
    phone_features: PhoneFeatures
    model_name: Optional[str] = "xgboost"

class PredictionResponse(BaseModel):
    product_id: str
    predicted_price: float
    model_used: str
    confidence_score: Optional[float] = 0.85
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    version: str
    performance_mae: float
    performance_r2: float

class PredictionHistory(BaseModel):
    id: int
    product_id: str
    predicted_price: float
    model_used: str
    created_at: datetime

    class Config:
        from_attributes = True