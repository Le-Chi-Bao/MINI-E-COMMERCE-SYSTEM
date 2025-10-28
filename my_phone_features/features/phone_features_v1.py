from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define entity
phone = Entity(
    name="phone",
    description="A smartphone product",
    join_keys=["product_id"],
)

# Data source từ file processed của bạn
phone_data_source = FileSource(
    name="phone_data_source",
    path="data/processed/phone_data_processed.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature View 1: Display Features
phone_display_fv = FeatureView(
    name="phone_display",
    entities=[phone],
    ttl=timedelta(days=365),
    schema=[
        Field(name="ScreenSize", dtype=Float32),
        Field(name="Res_Width", dtype=Float32),
        Field(name="Res_Height", dtype=Float32),
        Field(name="PPI", dtype=Float32),
        Field(name="total_resolution", dtype=Float32),
    ],
    source=phone_data_source,
    online=True,
    tags={"team": "display"},
)

# Feature View 2: Camera Features  
phone_camera_fv = FeatureView(
    name="phone_camera",
    entities=[phone],
    ttl=timedelta(days=365),
    schema=[
        Field(name="main_camera_mp", dtype=Float32),
        Field(name="num_cameras", dtype=Int64),
        Field(name="has_telephoto", dtype=Int64),
        Field(name="has_ultrawide", dtype=Int64),
        Field(name="has_ois", dtype=Int64),
        Field(name="camera_feature_count", dtype=Int64),
        Field(name="camera_score", dtype=Float32),
    ],
    source=phone_data_source,
    online=True,
    tags={"team": "camera"},
)

# Feature View 3: Product Info
phone_product_fv = FeatureView(
    name="phone_product",
    entities=[phone],
    ttl=timedelta(days=90),
    schema=[
        Field(name="NumberOfReview", dtype=Float32),
        Field(name="has_warranty", dtype=Int64),
    ],
    source=phone_data_source,
    online=True,
    tags={"team": "product"},
)