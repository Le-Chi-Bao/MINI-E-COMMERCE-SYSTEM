from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import time
import uuid

from app.database import get_db
from app import schemas, models
# from scripts.predictor import PhonePricePredictor
#################################################################################
# endpoints.py - Sửa import
import sys
import os
# Thêm đường dẫn đến scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from predictor import PhonePricePredictor  # ✅ IMPORT ĐÚNG


router = APIRouter(prefix="/api/v1", tags=["predictions"])
predictor = PhonePricePredictor()

@router.post("/predict", response_model=schemas.PredictionResponse)
async def predict_price(
    request: schemas.PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        features_dict = request.phone_features.dict()
        predicted_price = predictor.predict(features_dict, request.model_name)
        processing_time = time.time() - start_time
        product_id = f"phone_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        background_tasks.add_task(
            save_prediction_to_db,
            db=db,
            product_id=product_id,
            features=features_dict,
            predicted_price=predicted_price,
            model_used=request.model_name
        )
        
        return schemas.PredictionResponse(
            product_id=product_id,
            predicted_price=predicted_price,
            model_used=request.model_name,
            confidence_score=0.85,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def save_prediction_to_db(db: Session, product_id: str, features: dict, predicted_price: float, model_used: str):
    try:
        prediction = models.PhonePrediction(
            product_id=product_id,
            screen_size=features['screen_size'],
            resolution_width=features['resolution_width'],
            resolution_height=features['resolution_height'],
            main_camera_mp=features['main_camera_mp'],
            num_cameras=features['num_cameras'],
            has_telephoto=features['has_telephoto'],
            has_ultrawide=features['has_ultrawide'],
            has_ois=features['has_ois'],
            has_warranty=features['has_warranty'],
            number_of_reviews=features['number_of_reviews'],
            predicted_price=predicted_price,
            model_used=model_used
        )
        
        db.add(prediction)
        db.commit()
    except Exception as e:
        print(f"Failed to save prediction: {e}")
        db.rollback()

@router.get("/models", response_model=List[schemas.ModelInfo])
async def get_available_models(db: Session = Depends(get_db)):
    models_list = db.query(models.ModelVersion).filter(
        models.ModelVersion.is_active == True
    ).all()
    
    return [
        schemas.ModelInfo(
            model_name=model.model_name,
            version=model.version,
            performance_mae=model.performance_mae,
            performance_r2=model.performance_r2
        ) for model in models_list
    ]

@router.get("/predictions", response_model=List[schemas.PredictionHistory])
async def get_prediction_history(
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = db.query(models.PhonePrediction)\
        .order_by(models.PhonePrediction.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return predictions

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(predictor.models) > 0
    }