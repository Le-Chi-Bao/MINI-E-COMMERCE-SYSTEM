from feast import FeatureStore
import pandas as pd
import os

# Path đến Feast repo
feast_repo_path = "../my_phone_features"
fs = FeatureStore(repo_path=feast_repo_path)

print("📥 Preparing COMPLETE training data...")

# Load data gốc
source_path = "../my_phone_features/data/processed/phone_data_processed.parquet"
data = pd.read_parquet(source_path)

print(f"📊 Source data shape: {data.shape}")

# 🆕 TẤT CẢ FEATURES CHO 3 MODELS
all_features = [
    'ScreenSize', 'PPI', 'total_resolution',
    'camera_score', 'has_telephoto', 'has_ultrawide', 'popularity_score', 
    'value_score', 'price_segment',
    'has_warranty', 'NumberOfReview',
    'main_camera_mp', 'num_cameras', 'has_ois', 'camera_feature_count',
    'display_score'
]

# 🆕 TẤT CẢ TARGETS CHO 3 MODELS
all_targets = ['overall_score', 'is_premium', 'camera_rating']

# Kiểm tra features có tồn tại
available_features = [f for f in all_features if f in data.columns]
missing_features = [f for f in all_features if f not in data.columns]

available_targets = [t for t in all_targets if t in data.columns]
missing_targets = [t for t in all_targets if t not in data.columns]

print(f"✅ Available features: {len(available_features)}")
print(f"✅ Available targets: {available_targets}")

if missing_features:
    print(f"⚠️  Missing features: {missing_features}")
if missing_targets:
    print(f"❌ Missing targets: {missing_targets}")

# 🆕 TẠO TRAINING DATA VỚI TẤT CẢ FEATURES & TARGETS
training_data = data[available_features + available_targets + ['product_id']]

print(f"✅ Complete training data shape: {training_data.shape}")
print(f"🎯 Features: {len(available_features)}, Targets: {len(available_targets)}")

# Lưu training data
output_path = "../my_phone_features/data/training_data.parquet"
training_data.to_parquet(output_path, index=False)
print(f"💾 Saved: {output_path}")

print(f"\n📈 COMPLETE Training Data Summary:")
print(f"   Samples: {training_data.shape[0]}")
print(f"   Total columns: {training_data.shape[1]}")
print(f"   Features: {len(available_features)}")
print(f"   Targets: {available_targets}")

for target in available_targets:
    target_data = training_data[target]
    if target == 'is_premium':
        print(f"   {target}: {target_data.value_counts().to_dict()}")
    else:
        print(f"   {target}: {target_data.min():.1f} - {target_data.max():.1f}")

print(f"   🎉 Ready for ALL 3 models training!")