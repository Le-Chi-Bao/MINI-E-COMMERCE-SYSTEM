import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
    
    def load_raw_data(self):
        self.dataset = pd.read_csv(self.data_path)
        self.dataset['product_id'] = (self.dataset.index + 1).astype(str).str.zfill(3)
        print(f"📥 Loaded dataset with {len(self.dataset)} rows")
        return self
    
    def preprocess_for_feast(self, save_path=None):
        if self.dataset is None:
            raise ValueError("Please load data first using load_raw_data()")
        
        dataset_clean = self.dataset.drop([
            'Link', 'Name', 'Brand', 'DiscountedPercent', 'SoldQuantity', 'BatteryCapacity',
            'FrontCamera', 'GPU', 'ChargingPort', 'RAM', 'ROM', 'Rating',
            'Description', 'data_source'], axis=1)
        
        nan_count = dataset_clean.isnull().sum(axis=1)
        dataset_clean = dataset_clean[nan_count < 6]
        
        dataset_clean = self._add_timestamps(dataset_clean)
        
        if save_path:
            dataset_clean.to_parquet(save_path, index=False)
            print(f"💾 Saved processed data to {save_path}")
        
        return dataset_clean
    
    def _add_timestamps(self, df):
        df_processed = df.copy()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        random_days = np.random.randint(0, 365, len(df_processed))
        df_processed['event_timestamp'] = [start_date + timedelta(days=int(x)) for x in random_days]
        df_processed['created_timestamp'] = df_processed['event_timestamp']
        
        return df_processed
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        
        if self.dataset is None:
            raise ValueError("Please load data first")
        
        dataset_clean = self.preprocess_for_feast()
        
        X = dataset_clean.drop(['DiscountedPrice', 'product_id', 'event_timestamp', 'created_timestamp'], axis=1)
        y = dataset_clean['DiscountedPrice']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"🎯 Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    loader = DataLoader("../CrawlerData/final_data_phone.csv")
    loader.load_raw_data()
    feast_data = loader.preprocess_for_feast("data/processed/phone_data_processed.parquet")
    X_train, X_test, y_train, y_test = loader.get_train_test_split()