import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle
import json
from pathlib import Path
from datetime import datetime

from scripts.transformer import MobilePhoneTransformer, TargetTransformer

class ModelTrainer:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {
            'LinearRegression': LinearRegression(),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        
        self.results = {}
        self.feature_transformer = MobilePhoneTransformer()
        self.target_transformer = TargetTransformer()
    
    def train_from_dataframe(self, X_train, X_test, y_train, y_test):
        print("📊 Training from provided data...")
        
        X_train_processed = self.feature_transformer.fit_transform(X_train)
        X_test_processed = self.feature_transformer.transform(X_test)
        
        y_train_processed = self.target_transformer.fit_transform(y_train)
        y_test_processed = self.target_transformer.transform(y_test)
        
        self.feature_columns = X_train_processed.columns.tolist()
        
        print(f"🎯 Training with {X_train_processed.shape[1]} features")
        
        for name, model in self.models.items():
            print(f"\n🔹 Training {name}...")
            
            model.fit(X_train_processed, y_train_processed)
            y_pred = model.predict(X_test_processed)
            
            y_pred_original = self.target_transformer.inverse_transform(y_pred)
            y_test_original = self.target_transformer.inverse_transform(y_test_processed)
            
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)
            
            cv_scores = cross_val_score(
                model, X_train_processed, y_train_processed, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            cv_mae = np.mean(np.abs(cv_scores))
            
            self.results[name] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'cv_mae': cv_mae,
                'feature_importance': self._get_feature_importance(model, self.feature_columns)
            }
            
            print(f"   ✅ MAE: {mae:,.0f} VND")
            print(f"   ✅ R²: {r2:.4f}")
        
        self._save_models_and_results()
        self._print_comparison()
        
        return self.results
    
    def _get_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, model.coef_))
        return None
    
    def _save_models_and_results(self):
        for name, result in self.results.items():
            model_path = self.models_dir / f"{name.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        training_info = {
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_performance': {
                name: {
                    'mae': result['mae'],
                    'r2_score': result['r2'],
                    'cv_mae': result['cv_mae']
                } for name, result in self.results.items()
            },
            'best_model': min(self.results.items(), key=lambda x: x[1]['mae'])[0]
        }
        
        with open(self.models_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n💾 Models saved to {self.models_dir}/")
    
    def _print_comparison(self):
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*60)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mae'])
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"{i}. {name:15} | MAE: {result['mae']:>10,.0f} VND | R²: {result['r2']:>7.4f}")
        
        best_model = sorted_results[0][0]
        print(f"\n🏆 BEST MODEL: {best_model}")

if __name__ == "__main__":
    from scripts.data_loader import DataLoader
    
    loader = DataLoader("../CrawlerData/final_data_phone.csv")
    loader.load_raw_data()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    trainer = ModelTrainer()
    results = trainer.train_from_dataframe(X_train, X_test, y_train, y_test)