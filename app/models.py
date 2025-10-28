from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class PhonePrediction(Base):
    __tablename__ = "phone_predictions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(50), unique=True, index=True)
    
    # Phone features
    screen_size = Column(Float)
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    ppi = Column(Float)
    main_camera_mp = Column(Float)
    num_cameras = Column(Integer)
    has_telephoto = Column(Boolean)
    has_ultrawide = Column(Boolean)
    has_ois = Column(Boolean)
    has_warranty = Column(Boolean)
    number_of_reviews = Column(Float)
    
    # Prediction results
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    model_used = Column(String(50))
    confidence_score = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), index=True)
    version = Column(String(50))
    file_path = Column(String(255))
    performance_mae = Column(Float)
    performance_r2 = Column(Float)
    training_date = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    features_used = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())