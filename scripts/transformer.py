import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MobilePhoneTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.binary_map = {
            'has_telephoto': {'Không có camera tele': 0, 'Có camera tele': 1},
            'has_ultrawide': {'Không có camera siêu rộng': 0, 'Có camera siêu rộng': 1},
            'has_ois': {'Không có chống rung OIS': 0, 'Có chống rung OIS': 1},
            'has_warranty': {'Không có bảo hành': 0, 'Có bảo hành': 1}
        }
        
        self.noise_rules = {
            'ScreenSize': (4, 8),
            'NumberOfReview': (0, 1000),
            'main_camera_mp': (5, 200),
            'num_cameras': (1, 5),
            'Res_Width': (720, 3840),
            'Res_Height': (1280, 2160)
        }
        
        self.numeric_features = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        # Transform trước để có đầy đủ features
        X_temp = X.copy()
        X_temp = self._drop_unnecessary_columns(X_temp)
        X_temp = self._process_resolution(X_temp)
        X_temp = self._handle_binary_features(X_temp)
        X_temp = self._handle_noise_with_median(X_temp)
        X_temp = self._handle_missing_with_median(X_temp)
    
        # Sau đó mới fit scaler với đầy đủ features
        self.numeric_features = X_temp.select_dtypes(include=['float64']).columns.tolist()
    
        if len(self.numeric_features) > 0:
            self.scaler.fit(X_temp[self.numeric_features])
    
        self.feature_names_ = self.get_feature_names_out()
        return self
    
    def transform(self, X):
        X_processed = X.copy()
        
        X_processed = self._drop_unnecessary_columns(X_processed)
        X_processed = self._process_resolution(X_processed)
        X_processed = self._handle_binary_features(X_processed)
        X_processed = self._handle_noise_with_median(X_processed)
        X_processed = self._handle_missing_with_median(X_processed)
        X_processed = self._normalize_features(X_processed)
        X_processed = self._create_derived_features(X_processed)
        
        return X_processed
    
    def _drop_unnecessary_columns(self, df):
        columns_to_drop = ['is_new_product', 'has_original_accessories']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        return df.drop(columns=existing_columns)
    
    def _process_resolution(self, df):
        df_processed = df.copy()
        
        if 'Resolution' in df_processed.columns:
            resolution_split = df_processed['Resolution'].str.split('x', expand=True)
            df_processed[['Res_Width', 'Res_Height']] = resolution_split.astype('float64')
            
            width_height = df_processed[['Res_Width', 'Res_Height']]
            df_processed['Res_Width'], df_processed['Res_Height'] = \
                width_height.max(axis=1), width_height.min(axis=1)
            
            df_processed = df_processed.drop('Resolution', axis=1)
        
        return df_processed
    
    def _handle_binary_features(self, df):
        df_processed = df.copy()
        
        for feature, mapping in self.binary_map.items():
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].map(mapping).fillna(0).astype(int)
        
        return df_processed
    
    def _handle_noise_with_median(self, df):
        df_processed = df.copy()
        
        for feature, (low, high) in self.noise_rules.items():
            if feature in df_processed.columns:
                outliers_mask = (df_processed[feature] < low) | (df_processed[feature] > high)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    median_clean = df_processed[~outliers_mask][feature].median()
                    df_processed.loc[outliers_mask, feature] = median_clean
        
        return df_processed
    
    def _handle_missing_with_median(self, df):
        df_processed = df.copy()
        numeric_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for feature in numeric_features:
            if df_processed[feature].isnull().sum() > 0:
                median_value = df_processed[feature].median()
                df_processed[feature] = df_processed[feature].fillna(median_value)
        
        return df_processed
    
    def _normalize_features(self, df):
        df_processed = df.copy()
        current_numeric_features = df_processed.select_dtypes(include=['float64']).columns.tolist()
        
        if len(current_numeric_features) > 0:
            df_processed[current_numeric_features] = self.scaler.transform(
                df_processed[current_numeric_features]
            )
        
        return df_processed
    
    def _create_derived_features(self, df):
        df_processed = df.copy()
        
        if all(col in df_processed.columns for col in ['Res_Width', 'Res_Height', 'ScreenSize']):
            df_processed['PPI'] = np.sqrt(
                df_processed['Res_Width']**2 + df_processed['Res_Height']**2
            ) / df_processed['ScreenSize']
        
        if all(col in df_processed.columns for col in ['Res_Width', 'Res_Height']):
            df_processed['total_resolution'] = df_processed['Res_Width'] * df_processed['Res_Height']
        
        camera_features = ['has_telephoto', 'has_ultrawide', 'has_ois']
        if all(col in df_processed.columns for col in camera_features):
            df_processed['camera_feature_count'] = df_processed[camera_features].sum(axis=1)
        
        if all(col in df_processed.columns for col in ['main_camera_mp', 'num_cameras', 'camera_feature_count']):
            df_processed['camera_score'] = (
                df_processed['main_camera_mp'] * 0.4 +
                df_processed['num_cameras'] * 0.3 + 
                df_processed['camera_feature_count'] * 0.3
            )
        
        return df_processed
    
    def get_feature_names_out(self, input_features=None):
        return None

class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_transform=True, handle_outliers=True, outlier_threshold=70000000):
        self.log_transform = log_transform
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        
    def fit(self, y, X=None):
        return self
    
    def transform(self, y):
        y_processed = y.copy()
        
        y_processed = y_processed.replace('Giá Liên Hệ', np.nan)
        y_processed = pd.to_numeric(y_processed, errors='coerce')
        
        # THÊM XỬ LÝ NAN - QUAN TRỌNG!
        if y_processed.isnull().sum() > 0:
            print(f"⚠️  Found {y_processed.isnull().sum()} NaN values in target, filling with median")
            median_value = y_processed.median()
            y_processed = y_processed.fillna(median_value)
        
        if self.handle_outliers:
            outliers_mask = y_processed > self.outlier_threshold
            if outliers_mask.any():
                median_clean = y_processed[~outliers_mask].median()
                y_processed[outliers_mask] = median_clean
        
        if self.log_transform:
            y_processed = np.log1p(y_processed)
        
        return y_processed
    
    def inverse_transform(self, y):
        if self.log_transform:
            return np.expm1(y)
        return y

# Usage functions
def create_preprocessing_pipeline():
    feature_transformer = MobilePhoneTransformer()
    target_transformer = TargetTransformer()
    return feature_transformer, target_transformer

def preprocess_data(X_train, X_test, y_train, y_test):
    feature_transformer = MobilePhoneTransformer()
    target_transformer = TargetTransformer()
    
    X_train_processed = feature_transformer.fit_transform(X_train)
    X_test_processed = feature_transformer.transform(X_test)
    
    y_train_processed = target_transformer.fit_transform(y_train)
    y_test_processed = target_transformer.transform(y_test)
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train_reg': y_train_processed,
        'y_test_reg': y_test_processed,
        'feature_transformer': feature_transformer,
        'target_transformer': target_transformer
    }