import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import os

os.makedirs("../models", exist_ok=True)

print("🚀 Training All 3 Phone Prediction Models...")

# Load training data
data_path = "../my_phone_features/data/training_data.parquet"
data = pd.read_parquet(data_path)
print(f"📊 Training data: {data.shape}")

# ==================== MODEL 1: SMART RECOMMENDER ====================
print("\n🤖 1. Training Smart Recommender...")

recommender_features = [
    'ScreenSize', 'PPI', 'total_resolution',
    'camera_score', 'has_telephoto', 'has_ultrawide',
    'popularity_score', 'value_score', 'price_segment',
    'has_warranty', 'NumberOfReview'
]
recommender_target = 'overall_score'

print(f"   🎯 Predicting: {recommender_target}")
print(f"   📊 Features: {len(recommender_features)}")

# Prepare data
X_recom = data[recommender_features]
y_recom = data[recommender_target]

print(f"   📈 Target stats: min={y_recom.min():.1f}, max={y_recom.max():.1f}, mean={y_recom.mean():.1f}")

# Train/test split
X_train_recom, X_test_recom, y_train_recom, y_test_recom = train_test_split(
    X_recom, y_recom, test_size=0.2, random_state=42
)
print(f"   🔀 Train set: {X_train_recom.shape}, Test set: {X_test_recom.shape}")

# Chuẩn hóa features
scaler_recom = StandardScaler()
X_train_recom_scaled = scaler_recom.fit_transform(X_train_recom)
X_test_recom_scaled = scaler_recom.transform(X_test_recom)

# Train model
model_recom = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model_recom.fit(X_train_recom_scaled, y_train_recom)

# Evaluate
y_pred_recom = model_recom.predict(X_test_recom_scaled)
mse_recom = mean_squared_error(y_test_recom, y_pred_recom)
r2_recom = r2_score(y_test_recom, y_pred_recom)

print(f"   ✅ R²: {r2_recom:.3f}")
print(f"   📊 RMSE: {np.sqrt(mse_recom):.2f}")

# Save model
joblib.dump(model_recom, "../models/model_recommender.pkl")
joblib.dump(scaler_recom, "../models/scaler_recommender.pkl")
print("   💾 Saved: model_recommender.pkl")

# ==================== MODEL 2: VALUE DETECTOR ====================
print("\n💰 2. Training Value Detector...")

# Features cho value detection
value_features = [
    'value_score', 'price_segment', 'overall_score', 
    'display_score', 'camera_rating', 'PPI', 'ScreenSize',
    'camera_score', 'main_camera_mp', 'NumberOfReview'
]
value_target = 'is_premium'

# Filter available features
available_value_features = [f for f in value_features if f in data.columns]

X_value = data[available_value_features]
y_value = data[value_target]

print(f"   🎯 Predicting: {value_target}")
print(f"   📊 Features: {len(available_value_features)}")
print(f"   📈 Class balance: {y_value.value_counts().to_dict()}")

# Train/test split
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_value, y_value, test_size=0.2, random_state=42, stratify=y_value
)

# Chuẩn hóa
scaler_value = StandardScaler()
X_train_val_scaled = scaler_value.fit_transform(X_train_val)
X_test_val_scaled = scaler_value.transform(X_test_val)

# Train model
model_value = RandomForestClassifier(n_estimators=100, random_state=42)
model_value.fit(X_train_val_scaled, y_train_val)

# Evaluate
y_pred_val = model_value.predict(X_test_val_scaled)
accuracy = accuracy_score(y_test_val, y_pred_val)

print(f"   ✅ Accuracy: {accuracy:.3f}")
print(f"   📊 Classification Report:")
print(classification_report(y_test_val, y_pred_val))

# Save model
joblib.dump(model_value, "../models/model_value.pkl")
joblib.dump(scaler_value, "../models/scaler_value.pkl")
print("   💾 Saved: model_value.pkl")

# ==================== MODEL 3: CAMERA PREDICTOR ====================
print("\n📸 3. Training Camera Predictor...")

# Features cho camera prediction
camera_features = [
    'main_camera_mp', 'num_cameras', 'has_telephoto', 
    'has_ultrawide', 'has_ois', 'camera_feature_count',
    'PPI', 'total_resolution', 'ScreenSize',
    'value_score', 'is_premium', 'NumberOfReview'
]
camera_target = 'camera_rating'

# Filter available features
available_camera_features = [f for f in camera_features if f in data.columns]

X_camera = data[available_camera_features]
y_camera = data[camera_target]

print(f"   🎯 Predicting: {camera_target}")
print(f"   📊 Features: {len(available_camera_features)}")
print(f"   📈 Target range: {y_camera.min():.1f} - {y_camera.max():.1f}")

# Train/test split
X_train_cam, X_test_cam, y_train_cam, y_test_cam = train_test_split(
    X_camera, y_camera, test_size=0.2, random_state=42
)

# Chuẩn hóa
scaler_camera = StandardScaler()
X_train_cam_scaled = scaler_camera.fit_transform(X_train_cam)
X_test_cam_scaled = scaler_camera.transform(X_test_cam)

# Train model
model_camera = RandomForestRegressor(n_estimators=100, random_state=42)
model_camera.fit(X_train_cam_scaled, y_train_cam)

# Evaluate
y_pred_cam = model_camera.predict(X_test_cam_scaled)
r2_camera = r2_score(y_test_cam, y_pred_cam)

print(f"   ✅ R²: {r2_camera:.3f}")
print(f"   📊 RMSE: {np.sqrt(mean_squared_error(y_test_cam, y_pred_cam)):.2f}")

# Save model
joblib.dump(model_camera, "../models/model_camera.pkl")
joblib.dump(scaler_camera, "../models/scaler_camera.pkl")
print("   💾 Saved: model_camera.pkl")

print(f"\n🎉 ALL 3 MODELS TRAINED SUCCESSFULLY!")
print("   🤖 Smart Recommender - overall_score prediction")
print("   💰 Value Detector - is_premium classification") 
print("   📸 Camera Predictor - camera_rating prediction")

# Tạo thư mục models nếu chưa có
os.makedirs("../models", exist_ok=True)
print(f"\n📁 Models saved in: ../models/")