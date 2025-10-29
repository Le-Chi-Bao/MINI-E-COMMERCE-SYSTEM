# pipeline.py
import argparse
import sys
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description="🚀 Complete Phone Price Prediction Pipeline")
    parser.add_argument("--data-path", default="../Data/raw/final_data_phone.csv", 
                       help="Path to raw data file")
    parser.add_argument("--models-dir", default="../models", 
                       help="Directory to save trained models")
    parser.add_argument("--skip-feast", action="store_true", 
                       help="Skip Feast feature store setup")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip model training")
    args = parser.parse_args()
    
    print("=" * 70)
    print("📱 PHONE PRICE PREDICTION PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Loading and Preprocessing
        print("\n1️⃣  STEP 1: Loading and Preprocessing Data...")
        from data_loader import DataLoader
        
        loader = DataLoader(args.data_path)
        loader.load_raw_data()
        X_train, X_test, y_train, y_test = loader.get_train_test_split()
        
        # Save processed data for Feast
        if not args.skip_feast:
            loader.preprocess_for_feast("../my_phone_features/data/processed/phone_data_processed.parquet")
        
        print("✅ Data loading completed!")
        
        # Step 2: Feature Store Setup (Optional)
        if not args.skip_feast:
            print("\n2️⃣  STEP 2: Setting up Feature Store...")
            from feature_store import FeatureStoreManager
            
            fs_manager = FeatureStoreManager()
            fs_manager.setup_complete_pipeline()
            print("✅ Feature store setup completed!")
        else:
            print("\n⏭️  STEP 2: Skipping Feature Store setup")
        
        # Step 3: Model Training
        if not args.skip_training:
            print("\n3️⃣  STEP 3: Training Models...")
            from model_trainer import ModelTrainer
            
            trainer = ModelTrainer(models_dir=args.models_dir)
            results = trainer.train_from_dataframe(X_train, X_test, y_train, y_test)
            print("✅ Model training completed!")
        else:
            print("\n⏭️  STEP 3: Skipping model training")
        
        # Step 4: Model Validation and Prediction
        print("\n4️⃣  STEP 4: Model Validation and Prediction...")
        from predictor import PhonePricePredictor
        
        predictor = PhonePricePredictor(models_dir=args.models_dir)
        
        # Test prediction with sample data
        sample_phone = {
            'screen_size': 6.7,
            'resolution_width': 2796,
            'resolution_height': 1290,
            'main_camera_mp': 48,
            'num_cameras': 3,
            'has_telephoto': 1,
            'has_ultrawide': 1,
            'has_ois': 1,
            'has_warranty': 1,
            'number_of_reviews': 150
        }
        
        # Test all models
        model_performance = predictor.get_model_performance()
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("-" * 50)
        for model_name, metrics in model_performance.items():
            print(f"{model_name:20} | MAE: {metrics['mae']:>10,} VND | R²: {metrics['r2']:.3f}")
        
        # Make a sample prediction
        predicted_price = predictor.predict(sample_phone, model_name="xgboost")
        print(f"\n🎯 SAMPLE PREDICTION: {predicted_price:,.0f} VND")
        
        print("✅ Pipeline execution completed!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        sys.exit(1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()