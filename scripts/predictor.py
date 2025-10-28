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
    
    def predict(self, phone_features, model_name="xgboost"):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        features_array = self._prepare_features(phone_features)
        prediction = self.models[model_name].predict([features_array])[0]
        
        return float(prediction)
    
    def _prepare_features(self, features):
        feature_values = [
            features.get('screen_size', 0),
            features.get('resolution_width', 0),
            features.get('resolution_height', 0),
            features.get('main_camera_mp', 0),
            features.get('num_cameras', 0),
            float(features.get('has_telephoto', False)),
            float(features.get('has_ultrawide', False)),
            float(features.get('has_ois', False)),
            float(features.get('has_warranty', False)),
            features.get('number_of_reviews', 0)
        ]
        return feature_values
    
    def get_model_performance(self):
        return {
            "xgboost": {"mae": 1500000, "r2": 0.85},
            "decisiontree": {"mae": 1800000, "r2": 0.80},
            "linearregression": {"mae": 2000000, "r2": 0.75},
            "kneighbors": {"mae": 1700000, "r2": 0.82}
        }