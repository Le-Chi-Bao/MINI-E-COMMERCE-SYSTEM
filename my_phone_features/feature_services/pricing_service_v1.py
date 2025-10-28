from feast import FeatureService
from features.phone_features import (
    phone_display_fv,
    phone_camera_fv,
    phone_product_fv
)

pricing_service = FeatureService(
    name="pricing_service",
    features=[
        phone_display_fv,
        phone_camera_fv,
        phone_product_fv,
    ],
    tags={"purpose": "price_prediction"},
)