# 📁 scripts/train_model.py (THÊM MỚI)
import argparse
from scripts.data_loader import DataLoader
from scripts.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/final_data_phone.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    
    print("🚀 Starting model training...")
    
    # Load data
    loader = DataLoader(args.data_path)
    loader.load_raw_data()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    # Train models
    trainer = ModelTrainer(models_dir=args.models_dir)
    results = trainer.train_from_dataframe(X_train, X_test, y_train, y_test)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()