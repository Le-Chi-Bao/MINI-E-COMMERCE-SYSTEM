import pandas as pd
import joblib
import os
from feast import FeatureStore

class PhonePredictor:
    def __init__(self):
        # Load Feast store
        self.fs = FeatureStore(repo_path="../my_phone_features")
        
        # Load model và scaler
        self.model = joblib.load("../models/model_recommender.pkl")
        self.scaler = joblib.load("../models/scaler_recommender.pkl")
        
        # Features cần cho prediction (giống training)
        self.features = [
            'ScreenSize', 'PPI', 'total_resolution',
            'camera_score', 'has_telephoto', 'has_ultrawide',
            'camera_rating', 'popularity_score', 
            'value_score', 'price_segment',
            'has_warranty', 'NumberOfReview'
        ]
        
        self.feature_refs = [
            "phone_display:ScreenSize", "phone_display:PPI", "phone_display:total_resolution",
            "phone_camera:camera_score", "phone_camera:has_telephoto", "phone_camera:has_ultrawide",
            "phone_ratings:camera_rating", "phone_ratings:popularity_score",
            "phone_value:value_score", "phone_value:price_segment",
            "phone_product:has_warranty", "phone_product:NumberOfReview"
        ]
    
    def predict_phone_score(self, product_id):
        try:
            # Lấy features từ Feast
            feature_data = self.fs.get_online_features(
                entity_rows=[{"product_id": product_id}],
                features=self.feature_refs
            ).to_df()
            
            # 🆕 CHỈ CHỌN ĐÚNG 12 FEATURES ĐÃ TRAINING
            X_pred = feature_data[self.features]
            
            # Chuẩn hóa features
            features_scaled = self.scaler.transform(X_pred)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            return {
                'product_id': product_id,
                'predicted_score': round(prediction, 1),
                'actual_score': feature_data['overall_score'].iloc[0] if 'overall_score' in feature_data.columns else 'N/A',
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'product_id': product_id,
                'error': str(e),
                'status': 'error'
            }

# Test prediction
print("🚀 Testing Phone Prediction Service...")
predictor = PhonePredictor()

# Test với 3 điện thoại
test_phones = ["001", "050", "100"]
for phone_id in test_phones:
    result = predictor.predict_phone_score(phone_id)
    if result['status'] == 'success':
        actual_info = f", Actual Score = {result['actual_score']}" if result['actual_score'] != 'N/A' else ""
        print(f"📱 Phone {phone_id}: Predicted Score = {result['predicted_score']}{actual_info}")
    else:
        print(f"❌ Phone {phone_id}: Error - {result['error']}")

print("\n🎉 Prediction Service is ready!")