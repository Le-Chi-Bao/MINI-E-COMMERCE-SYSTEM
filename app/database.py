from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://admin:password123@localhost:3306/phone_predictor")
# DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://admin:password123@localhost:3306/phone_predictor")
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:120906@localhost:3306/phone_predictor")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    from app.models import Base
    Base.metadata.create_all(bind=engine)

def init_sample_data():
    from app.models import ModelVersion
    
    db = SessionLocal()
    
    try:
        existing_models = db.query(ModelVersion).count()
        if existing_models == 0:
            models = [
                ModelVersion(
                    model_name="xgboost",
                    version="v1.0",
                    file_path="models/xgboost.pkl",
                    performance_mae=1500000,
                    performance_r2=0.85,
                    training_date=datetime.now(),
                    is_active=True,
                    features_used='["screen_size", "resolution_width", "main_camera_mp", "num_cameras"]'
                ),
                ModelVersion(
                    model_name="decisiontree",
                    version="v1.0",
                    file_path="models/decisiontree.pkl",
                    performance_mae=1800000,
                    performance_r2=0.80,
                    training_date=datetime.now(),
                    is_active=True,
                    features_used='["screen_size", "resolution_width", "main_camera_mp", "num_cameras"]'
                ),
                ModelVersion(
                    model_name="linearregression",
                    version="v1.0",
                    file_path="models/linearregression.pkl",
                    performance_mae=2000000,
                    performance_r2=0.75,
                    training_date=datetime.now(),
                    is_active=True,
                    features_used='["screen_size", "resolution_width", "main_camera_mp", "num_cameras"]'
                ),
                ModelVersion(
                    model_name="kneighbors",
                    version="v1.0",
                    file_path="models/kneighbors.pkl",
                    performance_mae=1700000,
                    performance_r2=0.82,
                    training_date=datetime.now(),
                    is_active=True,
                    features_used='["screen_size", "resolution_width", "main_camera_mp", "num_cameras"]'
                )
            ]
            
            for model in models:
                db.add(model)
            
            db.commit()
            print("✅ Sample models initialized")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()