from feast import FeatureService
from features.phone_features import (
    phone_display_fv,
    phone_camera_fv,
    phone_product_fv,
    phone_ratings_fv,
    phone_value_fv,
)

# 🎯 1. SMART PHONE RECOMMENDER - Cho người dùng tìm điện thoại phù hợp
smart_recommender_service = FeatureService(
    name="smart_phone_recommender",
    features=[
        # Display quality - trải nghiệm người dùng
        phone_display_fv[["ScreenSize", "PPI", "total_resolution"]],
        
        # Camera capabilities - nhu cầu chụp ảnh
        phone_camera_fv[["camera_score", "has_telephoto", "has_ultrawide"]],
        
        # Ratings & popularity - đánh giá thực tế
        phone_ratings_fv[["overall_score", "camera_rating", "popularity_score"]],
        
        # Value for money - ngân sách hợp lý
        phone_value_fv[["value_score", "price_segment"]],
        
        # Product trust - bảo hành & đánh giá
        phone_product_fv[["has_warranty", "NumberOfReview"]]
    ],
    tags={"purpose": "recommendation", "team": "product", "latency": "medium"}
)

# 💰 2. VALUE FOR MONEY DETECTOR - Tìm điện thoại tốt nhất theo ngân sách
value_detector_service = FeatureService(
    name="value_for_money_detector", 
    features=[
        # Core value metrics
        phone_value_fv[["value_score", "is_premium", "price_segment"]],
        
        # Performance scores
        phone_ratings_fv[["overall_score", "display_score", "camera_rating"]],
        
        # Key specifications
        phone_display_fv[["PPI", "ScreenSize"]],
        phone_camera_fv[["camera_score", "main_camera_mp"]],
        
        # Social proof
        phone_product_fv[["NumberOfReview"]]
    ],
    tags={"purpose": "value_analysis", "team": "analytics", "latency": "fast"}
)

# 📸 3. CAMERA ENTHUSIAST PREDICTOR - Cho người dùng quan tâm camera
camera_enthusiast_service = FeatureService(
    name="camera_enthusiast_predictor",
    features=[
        # Comprehensive camera features
        phone_camera_fv,  # All camera features
        
        # Camera-specific ratings
        phone_ratings_fv[["camera_rating", "overall_score"]],
        
        # Display quality (ảnh hưởng đến xem ảnh)
        phone_display_fv[["PPI", "total_resolution", "ScreenSize"]],
        
        # Value consideration
        phone_value_fv[["value_score", "is_premium"]],
        
        # Popularity (độ tin cậy)
        phone_product_fv[["NumberOfReview"]]
    ],
    tags={"purpose": "camera_focused", "team": "camera", "latency": "medium"}
)