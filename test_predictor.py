# test_scaled_predictor.py
import sys
import os
import numpy as np

sys.path.append('scripts')
sys.path.append('app')

def main():
    print("🧪 TEST WITH SCALING & INVERSE TRANSFORM")
    print("=" * 60)
    
    from predictor import PhonePricePredictor
    
    predictor = PhonePricePredictor()
    
    print(f"✅ Models loaded: {len(predictor.models)}")
    
    # Test data với giá trị thực tế
    test_phones = [
        {
            'name': 'iPhone 15 Pro',
            'screen_size': 6.1,
            'resolution_width': 2556,
            'resolution_height': 1179,
            'main_camera_mp': 48,
            'num_cameras': 3,
            'has_telephoto': 1,
            'has_ultrawide': 1,
            'has_ois': 1,
            'has_warranty': 1,
            'number_of_reviews': 1000
        },
        {
            'name': 'Samsung Galaxy A54',
            'screen_size': 6.4,
            'resolution_width': 2340,
            'resolution_height': 1080,
            'main_camera_mp': 50,
            'num_cameras': 4,
            'has_telephoto': 0,
            'has_ultrawide': 1,
            'has_ois': 1,
            'has_warranty': 1,
            'number_of_reviews': 500
        }
    ]
    
    print("\n📊 PREDICTION RESULTS WITH FIXES:")
    print("=" * 50)
    
    for phone in test_phones:
        print(f"\n📱 {phone['name']}:")
        print("-" * 40)
        
        # Kiểm tra features trước và sau scaling
        raw_features = [
            phone['screen_size'], phone['resolution_width'], phone['resolution_height'],
            phone['main_camera_mp'], phone['num_cameras'], phone['has_telephoto'],
            phone['has_ultrawide'], phone['has_ois'], phone['has_warranty'],
            phone['number_of_reviews']
        ]
        
        print(f"Raw features - Screen: {phone['screen_size']}, Cam: {phone['main_camera_mp']}MP")
        
        # Test từng model
        for model_name in predictor.get_available_models():
            try:
                price = predictor.predict(phone, model_name)
                
                # Phân loại kết quả
                if price <= 0:
                    status = "❌ NEGATIVE"
                elif price < 1_000_000:
                    status = "⚠️ TOO LOW"
                elif price > 50_000_000:
                    status = "⚠️ TOO HIGH"
                else:
                    status = "✅ REASONABLE"
                
                print(f"  {model_name:18}: {price:>12,.0f} VND {status}")
                
            except Exception as e:
                print(f"  {model_name:18}: ❌ {str(e)[:40]}")

if __name__ == "__main__":
    main()