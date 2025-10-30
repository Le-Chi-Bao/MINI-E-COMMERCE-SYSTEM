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
        
        # 🆕 SỬA: CHỈ 11 FEATURES GIỐNG TRAINING (bỏ camera_rating)
        self.features = [
            'ScreenSize', 'PPI', 'total_resolution',
            'camera_score', 'has_telephoto', 'has_ultrawide',
            'popularity_score', 'value_score', 'price_segment',
            'has_warranty', 'NumberOfReview'
        ]
        
        self.feature_refs = [
            "phone_display:ScreenSize", "phone_display:PPI", "phone_display:total_resolution",
            "phone_camera:camera_score", "phone_camera:has_telephoto", "phone_camera:has_ultrawide",
            "phone_ratings:popularity_score",
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
            
            # 🆕 CHỈ CHỌN ĐÚNG 11 FEATURES ĐÃ TRAINING
            X_pred = feature_data[self.features]
            
            print(f"🔍 Features retrieved: {X_pred.shape}")  # Debug
            
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

# 🆕 SỬA: MultiModelPredictor với feature refs đúng
class MultiModelPredictor:
    def __init__(self):
        self.fs = FeatureStore(repo_path="../my_phone_features")
        
        # Load cả 3 models
        self.model_recom = joblib.load("../models/model_recommender.pkl")
        self.scaler_recom = joblib.load("../models/scaler_recommender.pkl")
        
        self.model_value = joblib.load("../models/model_value.pkl")
        self.scaler_value = joblib.load("../models/scaler_value.pkl")
        
        self.model_camera = joblib.load("../models/model_camera.pkl")
        self.scaler_camera = joblib.load("../models/scaler_camera.pkl")
        
        # 🆕 SỬA: Feature refs cố định cho từng model
        self.feature_refs_recom = [
            "phone_display:ScreenSize", "phone_display:PPI", "phone_display:total_resolution",
            "phone_camera:camera_score", "phone_camera:has_telephoto", "phone_camera:has_ultrawide",
            "phone_ratings:popularity_score",
            "phone_value:value_score", "phone_value:price_segment",
            "phone_product:has_warranty", "phone_product:NumberOfReview"
        ]
        
        self.feature_refs_value = [
            "phone_value:value_score", "phone_value:price_segment", 
            "phone_ratings:overall_score", "phone_ratings:display_score",
            "phone_ratings:camera_rating", "phone_display:PPI", "phone_display:ScreenSize",
            "phone_camera:camera_score", "phone_camera:main_camera_mp",
            "phone_product:NumberOfReview"
        ]
        
        self.feature_refs_camera = [
            "phone_camera:main_camera_mp", "phone_camera:num_cameras", 
            "phone_camera:has_telephoto", "phone_camera:has_ultrawide", 
            "phone_camera:has_ois", "phone_camera:camera_feature_count",
            "phone_display:PPI", "phone_display:total_resolution", "phone_display:ScreenSize",
            "phone_value:value_score", "phone_value:is_premium", 
            "phone_product:NumberOfReview"
        ]
        
        # Feature mapping
        self.features_recom = [
            'ScreenSize', 'PPI', 'total_resolution', 'camera_score', 
            'has_telephoto', 'has_ultrawide', 'popularity_score', 
            'value_score', 'price_segment', 'has_warranty', 'NumberOfReview'
        ]
        
        self.features_value = [
            'value_score', 'price_segment', 'overall_score', 'display_score', 
            'camera_rating', 'PPI', 'ScreenSize', 'camera_score', 
            'main_camera_mp', 'NumberOfReview'
        ]
        
        self.features_camera = [
            'main_camera_mp', 'num_cameras', 'has_telephoto', 'has_ultrawide', 
            'has_ois', 'camera_feature_count', 'PPI', 'total_resolution', 
            'ScreenSize', 'value_score', 'is_premium', 'NumberOfReview'
        ]
    
    def predict_all(self, product_id):
        try:
            # Lấy features cho từng model
            feature_data_recom = self.fs.get_online_features(
                entity_rows=[{"product_id": product_id}],
                features=self.feature_refs_recom
            ).to_df()
            
            feature_data_value = self.fs.get_online_features(
                entity_rows=[{"product_id": product_id}],
                features=self.feature_refs_value
            ).to_df()
            
            feature_data_camera = self.fs.get_online_features(
                entity_rows=[{"product_id": product_id}],
                features=self.feature_refs_camera
            ).to_df()
            
            # Predict từng model
            results = {}
            
            # Model 1: Smart Recommender
            X_recom = feature_data_recom[self.features_recom]
            X_recom_scaled = self.scaler_recom.transform(X_recom)
            results['overall_score'] = round(self.model_recom.predict(X_recom_scaled)[0], 1)
            
            # Model 2: Value Detector
            X_value = feature_data_value[self.features_value]
            X_value_scaled = self.scaler_value.transform(X_value)
            results['is_premium'] = int(self.model_value.predict(X_value_scaled)[0])
            results['premium_prob'] = round(self.model_value.predict_proba(X_value_scaled)[0][1], 3)
            
            # Model 3: Camera Predictor
            X_camera = feature_data_camera[self.features_camera]
            X_camera_scaled = self.scaler_camera.transform(X_camera)
            results['camera_rating'] = round(self.model_camera.predict(X_camera_scaled)[0], 1)
            
            return {
                'product_id': product_id,
                'predictions': results,
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
multi_predictor = MultiModelPredictor()

# Test với 3 điện thoại
test_phones = ["001", "050", "100"]

print("\n📱 SINGLE MODEL PREDICTION (Smart Recommender):")
for phone_id in test_phones:
    result = predictor.predict_phone_score(phone_id)
    if result['status'] == 'success':
        actual_info = f", Actual Score = {result['actual_score']}" if result['actual_score'] != 'N/A' else ""
        print(f"   Phone {phone_id}: Predicted Score = {result['predicted_score']}{actual_info}")
    else:
        print(f"   ❌ Phone {phone_id}: Error - {result['error']}")

print("\n🎯 MULTI-MODEL PREDICTION (All 3 Models):")
for phone_id in test_phones:
    result = multi_predictor.predict_all(phone_id)
    if result['status'] == 'success':
        preds = result['predictions']
        print(f"   Phone {phone_id}:")
        print(f"      🤖 Overall Score: {preds['overall_score']}")
        print(f"      💰 Premium: {preds['is_premium']} (prob: {preds['premium_prob']})")
        print(f"      📸 Camera Rating: {preds['camera_rating']}")
    else:
        print(f"   ❌ Phone {phone_id}: Error - {result['error']}")

print("\n🎉 Prediction Service is ready!")