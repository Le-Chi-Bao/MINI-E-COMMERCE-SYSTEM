# predictor.py
import pickle
import numpy as np
from pathlib import Path

class PhonePricePredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        try:
            for model_file in self.models_dir.glob("*.pkl"):
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            print(f"✅ Loaded {len(self.models)} models")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict(self, phone_features, model_name="kneighbors"):  # Đổi default thành model tốt nhất
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        features_array = self._prepare_features(phone_features)
        prediction = self.models[model_name].predict([features_array])[0]
        
        # ÁP DỤNG INVERSE TRANSFORM CHO PREDICTION
        # Vì target được log-transform trong training (theo transformer.py)
        prediction = self._inverse_transform_target(prediction)
        
        return float(prediction)
    
    def _prepare_features(self, features):
        # Lấy features cơ bản
        screen_size = features.get('screen_size', 0)
        resolution_width = features.get('resolution_width', 0)
        resolution_height = features.get('resolution_height', 0)
        main_camera_mp = features.get('main_camera_mp', 0)
        num_cameras = features.get('num_cameras', 0)
        has_telephoto = float(features.get('has_telephoto', False))
        has_ultrawide = float(features.get('has_ultrawide', False))
        has_ois = float(features.get('has_ois', False))
        has_warranty = float(features.get('has_warranty', False))
        number_of_reviews = features.get('number_of_reviews', 0)
        
        # TÍNH TOÁN DERIVED FEATURES
        ppi = np.sqrt(resolution_width**2 + resolution_height**2) / screen_size if screen_size > 0 else 0
        total_resolution = resolution_width * resolution_height
        camera_feature_count = has_telephoto + has_ultrawide + has_ois
        camera_score = (main_camera_mp * 0.4 + num_cameras * 0.3 + camera_feature_count * 0.3)
        
        # TẤT CẢ 14 FEATURES
        feature_values = [
            screen_size, resolution_width, resolution_height,
            main_camera_mp, num_cameras, has_telephoto,
            has_ultrawide, has_ois, has_warranty,
            number_of_reviews, ppi, total_resolution,
            camera_feature_count, camera_score
        ]
        
        # ÁP DỤNG FEATURE SCALING (giống như trong training)
        features_scaled = self._scale_features(feature_values)
        
        return features_scaled
    
    def _scale_features(self, features):
        """
        Scale features tương tự như trong MobilePhoneTransformer
        Đơn giản hóa: sử dụng scaling manual vì không có fitted scaler
        """
        features_array = np.array(features)
        
        # Manual scaling - bạn nên lưu scaler từ training để dùng ở đây
        # Tạm thời dùng scaling đơn giản
        mean_vals = np.mean(features_array)
        std_vals = np.std(features_array)
        
        if std_vals > 0:
            features_scaled = (features_array - mean_vals) / std_vals
        else:
            features_scaled = features_array
            
        return features_scaled.tolist()
    
    def _inverse_transform_target(self, y_pred):
        """
        Inverse transform cho target prediction
        Theo TargetTransformer trong transformer.py: log_transform=True
        """
        # Áp dụng expm1 để reverse log1p transformation
        return np.expm1(y_pred)
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def get_model_performance(self):
        # Cập nhật theo kết quả training thực tế
        return {
            "kneighbors": {"mae": 5593963, "r2": 0.2321},
            "linearregression": {"mae": 5851469, "r2": 0.1648},
            "xgboost": {"mae": 7080538, "r2": -0.3885},
            "decisiontree": {"mae": 7310993, "r2": -0.3869}
        }