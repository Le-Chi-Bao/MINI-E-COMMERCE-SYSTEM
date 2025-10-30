import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime, timedelta
import os

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
        
        self.numeric_features_ = None
        self.median_values_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        Fit transformer to data
        """
        X_temp = X.copy()
        
        # Fix data types first
        X_temp = self._ensure_numeric_types(X_temp)
        
        # Basic preprocessing without feature engineering
        X_temp = self._drop_unnecessary_columns(X_temp)
        X_temp = self._process_resolution(X_temp)
        X_temp = self._handle_binary_features(X_temp)
        
        # Identify numeric features and store median values
        self.numeric_features_ = X_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.median_values_ = {}
        
        for feature in self.numeric_features_:
            self.median_values_[feature] = X_temp[feature].median()
        
        # Fit scaler with numeric features (for potential future use)
        if len(self.numeric_features_) > 0:
            self.scaler.fit(X_temp[self.numeric_features_])
        
        return self
    
    def transform(self, X):
        """
        Transform input data - CREATE ALL FEATURES FOR 5 FEATURE VIEWS
        """
        X_processed = X.copy()
        
        # Fix data types first
        X_processed = self._ensure_numeric_types(X_processed)
        
        # Basic preprocessing WITHOUT normalization
        X_processed = self._basic_preprocessing_without_normalize(X_processed)
        
        # Feature engineering - create all derived features
        X_processed = self._create_all_features(X_processed)
        
        return X_processed
    
    def _ensure_numeric_types(self, df):
        """Ensure numeric columns have correct data type"""
        df_processed = df.copy()
        
        numeric_columns = [
            'ScreenSize', 'NumberOfReview', 'main_camera_mp', 'num_cameras',
            'Res_Width', 'Res_Height', 'DiscountedPrice'
        ]
        
        for col in numeric_columns:
            if col in df_processed.columns:
                # Special handling for DiscountedPrice
                if col == 'DiscountedPrice':
                    df_processed[col] = df_processed[col].replace('Giá Liên Hệ', np.nan)
                
                # Convert to numeric, errors -> NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    def _basic_preprocessing_without_normalize(self, df):
        """Apply basic preprocessing steps WITHOUT normalization"""
        df_processed = df.copy()
        
        df_processed = self._drop_unnecessary_columns(df_processed)
        df_processed = self._process_resolution(df_processed)
        df_processed = self._handle_binary_features(df_processed)
        df_processed = self._handle_missing_values(df_processed)
        df_processed = self._handle_outliers(df_processed)
        # INTENTIONALLY SKIP NORMALIZATION
        
        return df_processed
    
    def _drop_unnecessary_columns(self, df):
        """Drop unnecessary columns"""
        columns_to_drop = ['is_new_product', 'has_original_accessories']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        return df.drop(columns=existing_columns)
    
    def _process_resolution(self, df):
        """Process resolution column into width and height"""
        df_processed = df.copy()
        
        if 'Resolution' in df_processed.columns:
            resolution_split = df_processed['Resolution'].str.split('x', expand=True)
            df_processed[['Res_Width', 'Res_Height']] = resolution_split.astype('float64')
            
            # Ensure width is always larger than height
            width_height = df_processed[['Res_Width', 'Res_Height']]
            df_processed['Res_Width'], df_processed['Res_Height'] = \
                width_height.max(axis=1), width_height.min(axis=1)
            
            df_processed = df_processed.drop('Resolution', axis=1)
        
        return df_processed
    
    def _handle_binary_features(self, df):
        """Convert binary features to 0/1"""
        df_processed = df.copy()
        
        for feature, mapping in self.binary_map.items():
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].map(mapping).fillna(0).astype(int)
        
        return df_processed
    
    def _handle_missing_values(self, df):
        """Handle missing values using stored median values"""
        df_processed = df.copy()
        
        if self.median_values_ is not None:
            for feature, median_value in self.median_values_.items():
                if feature in df_processed.columns:
                    df_processed[feature] = df_processed[feature].fillna(median_value)
        
        return df_processed
    
    def _handle_outliers(self, df):
        """Handle outliers using stored median values"""
        df_processed = df.copy()
        
        if self.median_values_ is not None:
            for feature, (low, high) in self.noise_rules.items():
                if feature in df_processed.columns and feature in self.median_values_:
                    if pd.api.types.is_numeric_dtype(df_processed[feature]):
                        outliers_mask = (df_processed[feature] < low) | (df_processed[feature] > high)
                        outliers_count = outliers_mask.sum()
                        
                        if outliers_count > 0:
                            median_value = self.median_values_[feature]
                            df_processed.loc[outliers_mask, feature] = median_value
        
        return df_processed
    
    def _create_all_features(self, df):
        """
        Create all features needed for 5 Feature Views
        """
        df_processed = df.copy()
        
        # ==================== DISPLAY VIEW FEATURES ====================
        # Pixels Per Inch
        if all(col in df_processed.columns for col in ['Res_Width', 'Res_Height', 'ScreenSize']):
            valid_screen = df_processed['ScreenSize'] > 0
            df_processed.loc[valid_screen, 'PPI'] = np.sqrt(
                df_processed.loc[valid_screen, 'Res_Width']**2 + 
                df_processed.loc[valid_screen, 'Res_Height']**2
            ) / df_processed.loc[valid_screen, 'ScreenSize']
            df_processed['PPI'] = df_processed['PPI'].fillna(0)
        
        # Total resolution
        if all(col in df_processed.columns for col in ['Res_Width', 'Res_Height']):
            df_processed['total_resolution'] = df_processed['Res_Width'] * df_processed['Res_Height']
        
        # ==================== CAMERA VIEW FEATURES ====================
        # Camera feature count
        camera_features = ['has_telephoto', 'has_ultrawide', 'has_ois']
        existing_camera_features = [f for f in camera_features if f in df_processed.columns]
        if existing_camera_features:
            df_processed['camera_feature_count'] = df_processed[existing_camera_features].sum(axis=1)
        
        # Camera score
        if all(col in df_processed.columns for col in ['main_camera_mp', 'num_cameras', 'camera_feature_count']):
            df_processed['camera_score'] = (
                df_processed['main_camera_mp'] * 0.4 +
                df_processed['num_cameras'] * 0.3 + 
                df_processed['camera_feature_count'] * 0.3
            )
        
        # ==================== RATINGS VIEW FEATURES ====================
        # Camera rating (1-5 scale)
        if all(col in df_processed.columns for col in ['main_camera_mp', 'num_cameras', 'camera_feature_count']):
            main_camera_clean = df_processed['main_camera_mp'].fillna(0)
            num_cameras_clean = df_processed['num_cameras'].fillna(1)
            feature_count_clean = df_processed['camera_feature_count'].fillna(0)
            
            max_camera_mp = max(main_camera_clean.max(), 50.0)
            max_num_cameras = max(num_cameras_clean.max(), 5.0)
            max_features = max(feature_count_clean.max(), 3.0)
            
            camera_quality = (
                (main_camera_clean / max_camera_mp).clip(0, 1) * 0.4 +
                (num_cameras_clean / max_num_cameras).clip(0, 1) * 0.3 +
                (feature_count_clean / max_features).clip(0, 1) * 0.3
            )
            df_processed['camera_rating'] = (camera_quality * 4 + 1).round(1)
        
        # Display performance score
        if all(col in df_processed.columns for col in ['PPI', 'total_resolution', 'ScreenSize']):
            ppi_clean = df_processed['PPI'].fillna(300)
            resolution_clean = df_processed['total_resolution'].fillna(2000000)
            screen_clean = df_processed['ScreenSize'].fillna(6.0)
            
            ppi_90 = float(ppi_clean.quantile(0.9)) if len(ppi_clean) > 0 else 600
            resolution_90 = float(resolution_clean.quantile(0.9)) if len(resolution_clean) > 0 else 8000000
            screen_90 = float(screen_clean.quantile(0.9)) if len(screen_clean) > 0 else 7.5
            
            display_score = (
                (ppi_clean / ppi_90).clip(0, 1) * 40 +
                (resolution_clean / resolution_90).clip(0, 1) * 40 +
                (screen_clean / screen_90).clip(0, 1) * 20
            )
            df_processed['display_score'] = display_score.round(1)
        
        # Popularity score
        if 'NumberOfReview' in df_processed.columns:
            reviews_clean = df_processed['NumberOfReview'].fillna(0).clip(lower=0)
            current_max = float(reviews_clean.max())
            if current_max > 0:
                df_processed['popularity_score'] = (
                    (np.log1p(reviews_clean) / np.log1p(current_max)) * 100
                ).round(1)
            else:
                df_processed['popularity_score'] = 0
        
        # Overall score
        score_components = []
        if 'camera_rating' in df_processed.columns:
            camera_component = (df_processed['camera_rating'] - 1) / 4 * 100 * 0.3
            score_components.append(camera_component)
        
        if 'display_score' in df_processed.columns:
            display_component = df_processed['display_score'] * 0.4
            score_components.append(display_component)
        
        if 'popularity_score' in df_processed.columns:
            popularity_component = df_processed['popularity_score'] * 0.3
            score_components.append(popularity_component)
        
        if score_components:
            total_score = sum(score_components)
            df_processed['overall_score'] = total_score.round(1)
        else:
            df_processed['overall_score'] = 0
        
        # ==================== VALUE VIEW FEATURES ====================
        # Premium detector
        if 'DiscountedPrice' in df_processed.columns:
            price_clean = df_processed['DiscountedPrice'].fillna(8000000)
            df_processed['is_premium'] = (price_clean > 15000000).astype(int)
            
            # Price segment (0: budget, 1: mid_range, 2: premium)
            conditions = [
                price_clean <= 8000000,
                price_clean <= 15000000, 
                price_clean > 15000000
            ]
            choices = [0, 1, 2]
            df_processed['price_segment'] = np.select(conditions, choices, default=1)
        
        # Value for money score
        if all(col in df_processed.columns for col in ['DiscountedPrice', 'camera_score', 'display_score']):
            camera_max = max(float(df_processed['camera_score'].max()), 1.0)
            feature_value = (
                (df_processed['camera_score'].fillna(0) / camera_max * 50) + 
                (df_processed['display_score'].fillna(0) / 100 * 50)
            ).clip(0, 100)
            
            price_clean = df_processed['DiscountedPrice'].fillna(8000000)
            price_in_millions = price_clean / 1000000
            valid_price_mask = price_in_millions > 0.1
            
            value_scores = np.zeros(len(df_processed))
            if valid_price_mask.any():
                value_scores[valid_price_mask] = (
                    feature_value[valid_price_mask] / price_in_millions[valid_price_mask]
                ).round(2)
            
            if value_scores.max() > 0:
                value_scores = (value_scores / value_scores.max() * 10).round(2)
            
            df_processed['value_score'] = value_scores
        
        # Fill any remaining NaN values with 0
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
        
        return df_processed

    def get_feature_names_out(self, input_features=None):
        """
        Get all feature names for the 5 Feature Views
        """
        all_features = [
            # Display Features
            'ScreenSize', 'Res_Width', 'Res_Height', 'PPI', 'total_resolution',
            # Camera Features
            'main_camera_mp', 'num_cameras', 'has_telephoto', 'has_ultrawide', 
            'has_ois', 'camera_feature_count', 'camera_score',
            # Product Features
            'NumberOfReview', 'has_warranty',
            # Rating Features
            'camera_rating', 'display_score', 'popularity_score', 'overall_score',
            # Value Features
            'value_score', 'is_premium', 'price_segment'
        ]
        return [f for f in all_features if f in self.feature_names_] if hasattr(self, 'feature_names_') else all_features


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_transform=True, handle_outliers=True, outlier_threshold=70000000):
        self.log_transform = log_transform
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.median_value_ = None
        
    def fit(self, y, X=None):
        y_processed = self._preprocess_target(y)
        
        if self.handle_outliers:
            clean_y = y_processed[y_processed <= self.outlier_threshold]
            if len(clean_y) > 0:
                self.median_value_ = clean_y.median()
            else:
                self.median_value_ = y_processed.median()
        else:
            self.median_value_ = y_processed.median()
            
        return self
    
    def transform(self, y):
        y_processed = self._preprocess_target(y)
        
        if self.handle_outliers and self.median_value_ is not None:
            outliers_mask = y_processed > self.outlier_threshold
            if outliers_mask.any():
                y_processed[outliers_mask] = self.median_value_
        
        if self.log_transform:
            y_processed = np.log1p(y_processed)
        
        return y_processed
    
    def _preprocess_target(self, y):
        y_processed = y.copy()
        y_processed = y_processed.replace('Giá Liên Hệ', np.nan)
        y_processed = pd.to_numeric(y_processed, errors='coerce')
        
        if y_processed.isnull().sum() > 0:
            current_median = y_processed.median()
            y_processed = y_processed.fillna(current_median)
        
        return y_processed
    
    def inverse_transform(self, y):
        if self.log_transform:
            return np.expm1(y)
        return y


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


def create_feast_processed_data(raw_data_path, output_path, add_timestamps=True):
    """
    Transform raw data and save as processed data for Feast
    """
    # 1. Load raw data
    print("📥 Loading raw data...")
    raw_data = pd.read_csv(raw_data_path)
    print(f"   Raw data shape: {raw_data.shape}")
    
    # 2. Add product_id
    raw_data['product_id'] = (raw_data.index + 1).astype(str).str.zfill(3)
    
    # 3. Transform with all new features
    print("🔄 Transforming data...")
    transformer = MobilePhoneTransformer()
    
    try:
        transformed_data = transformer.fit_transform(raw_data)
        print(f"   Transformed data shape: {transformed_data.shape}")
        
    except Exception as e:
        print(f"❌ Transform error: {e}")
        raise e
    
    # 4. Add timestamps for Feast
    if add_timestamps:
        print("⏰ Adding timestamps for Feast...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        random_days = np.random.randint(0, 365, len(transformed_data))
        transformed_data['event_timestamp'] = [start_date + timedelta(days=int(x)) for x in random_days]
        transformed_data['created_timestamp'] = transformed_data['event_timestamp']
    
    # 5. Save transformed data
    print("💾 Saving processed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_data.to_parquet(output_path, index=False)
    
    # 6. Show results
    print(f"✅ Saved processed data: {output_path}")
    print(f"📊 Shape: {transformed_data.shape}")
    
    # Show important features
    important_features = ['PPI', 'camera_score', 'camera_rating', 'display_score', 
                         'overall_score', 'value_score', 'is_premium']
    print("\n🔍 Important features:")
    for feature in important_features:
        if feature in transformed_data.columns:
            stats = transformed_data[feature]
            print(f"   {feature}: {stats.min():.1f} - {stats.max():.1f} (mean: {stats.mean():.1f})")
    
    return transformed_data


def validate_processed_data(file_path):
    """
    Validate processed data has all required features
    """
    print(f"🔍 Validating {file_path}...")
    data = pd.read_parquet(file_path)
    
    required_features = [
        'product_id', 'ScreenSize', 'Res_Width', 'Res_Height', 'PPI',
        'main_camera_mp', 'num_cameras', 'camera_score', 'camera_rating',
        'display_score', 'overall_score', 'value_score', 'is_premium'
    ]
    
    print(f"📊 Data shape: {data.shape}")
    print("✅ Available features:")
    for feature in required_features:
        if feature in data.columns:
            print(f"   ✅ {feature}")
        else:
            print(f"   ❌ {feature} - MISSING!")
    
    # Check timestamps
    if 'event_timestamp' in data.columns and 'created_timestamp' in data.columns:
        print("✅ Timestamps: available")
    else:
        print("❌ Timestamps: missing")
    
    return data


if __name__ == "__main__":
    print("🚀 Mobile Phone Transformer - Feast Data Preparation")
    
    # Create processed data for Feast
    raw_path = "../Data/raw/final_data_phone.csv"
    output_path = "../my_phone_features/data/processed/phone_data_processed.parquet"
    
    try:
        # Create and save processed data
        processed_data = create_feast_processed_data(raw_path, output_path)
        
        print("\n" + "="*50)
        print("🎉 PROCESSED DATA READY FOR FEAST!")
        print("="*50)
        
        # Validate data
        validate_processed_data(output_path)
        
        # Show detailed stats
        print(f"\n📊 DETAILED STATS:")
        stats_data = pd.read_parquet(output_path)
        
        for col in ['ScreenSize', 'NumberOfReview', 'main_camera_mp', 'DiscountedPrice', 
                   'PPI', 'camera_score', 'overall_score', 'value_score']:
            if col in stats_data.columns:
                col_data = stats_data[col]
                print(f"   {col:15}: min={col_data.min():8.2f}, max={col_data.max():8.2f}, mean={col_data.mean():6.2f}")
        
        print(f"\n📝 Next steps:")
        print("1. Run: feast apply")
        print("2. Run: feast materialize 2023-01-01T00:00:00 2024-01-01T00:00:00")
        print("3. Test with: python test_features.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check file paths and data format")