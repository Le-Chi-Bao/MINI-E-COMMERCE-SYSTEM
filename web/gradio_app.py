# web/gradio_app.py
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
import joblib
from feast import FeatureStore

print("🚀 Loading Phone Prediction Models...")

class MultiModelPredictor:
    def __init__(self):
        try:
            # Load Feast store
            self.fs = FeatureStore(repo_path="../my_phone_features")
            
            # Load cả 3 models
            self.model_recom = joblib.load("../models/model_recommender.pkl")
            self.scaler_recom = joblib.load("../models/scaler_recommender.pkl")
            
            self.model_value = joblib.load("../models/model_value.pkl")
            self.scaler_value = joblib.load("../models/scaler_value.pkl")
            
            self.model_camera = joblib.load("../models/model_camera.pkl")
            self.scaler_camera = joblib.load("../models/scaler_camera.pkl")
            
            # Feature refs cho từng model
            self.feature_refs_recom = [
                "phone_display:ScreenSize", "phone_display:PPI", "phone_display:total_resolution",
                "phone_camera:camera_score", "phone_camera:has_telephoto", "phone_camera:has_ultrawide",
                "phone_ratings:popularity_score",
                "phone_value:value_score", "phone_value:price_segment",
                "phone_product:has_warranty", "phone_product:NumberOfReview"
            ]
            
            self.feature_refs_value = [
                "phone_value:value_score", "phone_value:price_segment", 
                "phone_ratings:overall_score", "phone_ratings:display_score",
                "phone_ratings:camera_rating", "phone_display:PPI", "phone_display:ScreenSize",
                "phone_camera:camera_score", "phone_camera:main_camera_mp",
                "phone_product:NumberOfReview"
            ]
            
            self.feature_refs_camera = [
                "phone_camera:main_camera_mp", "phone_camera:num_cameras", 
                "phone_camera:has_telephoto", "phone_camera:has_ultrawide", 
                "phone_camera:has_ois", "phone_camera:camera_feature_count",
                "phone_display:PPI", "phone_display:total_resolution", "phone_display:ScreenSize",
                "phone_value:value_score", "phone_value:is_premium", 
                "phone_product:NumberOfReview"
            ]
            
            # Feature mapping
            self.features_recom = [
                'ScreenSize', 'PPI', 'total_resolution', 'camera_score', 
                'has_telephoto', 'has_ultrawide', 'popularity_score', 
                'value_score', 'price_segment', 'has_warranty', 'NumberOfReview'
            ]
            
            self.features_value = [
                'value_score', 'price_segment', 'overall_score', 'display_score', 
                'camera_rating', 'PPI', 'ScreenSize', 'camera_score', 
                'main_camera_mp', 'NumberOfReview'
            ]
            
            self.features_camera = [
                'main_camera_mp', 'num_cameras', 'has_telephoto', 'has_ultrawide', 
                'has_ois', 'camera_feature_count', 'PPI', 'total_resolution', 
                'ScreenSize', 'value_score', 'is_premium', 'NumberOfReview'
            ]
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict_from_features(self, services: List[str], manual_features: Dict):
        """Dự đoán từ manual features"""
        try:
            results = {}
            
            # Model 1: Smart Recommender
            if "recommender" in services:
                X_recom = pd.DataFrame([manual_features])[self.features_recom]
                X_recom_scaled = self.scaler_recom.transform(X_recom)
                results['overall_score'] = round(self.model_recom.predict(X_recom_scaled)[0], 1)
            
            # Model 2: Value Detector
            if "value_detector" in services:
                X_value = pd.DataFrame([manual_features])[self.features_value]
                X_value_scaled = self.scaler_value.transform(X_value)
                results['is_premium'] = int(self.model_value.predict(X_value_scaled)[0])
                results['premium_probability'] = round(self.model_value.predict_proba(X_value_scaled)[0][1], 3)
            
            # Model 3: Camera Predictor
            if "camera_predictor" in services:
                X_camera = pd.DataFrame([manual_features])[self.features_camera]
                X_camera_scaled = self.scaler_camera.transform(X_camera)
                results['camera_rating'] = round(self.model_camera.predict(X_camera_scaled)[0], 1)
            
            return {
                'predictions': results,
                'status': 'success',
                'services_used': services
            }
            
        except Exception as e:
            return {
                'error': f"Prediction error: {str(e)}",
                'status': 'error'
            }

# ==================== VISUALIZATION ====================

def create_visualizations(predictions):
    """Tạo biểu đồ trực quan cho 3 features chính"""
    viz_figures = []
    
    # 1. Overall Score Gauge Chart
    if 'overall_score' in predictions:
        score = predictions['overall_score']
        overall_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ĐIỂM TỔNG QUAN", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        overall_fig.update_layout(height=300, margin=dict(t=50, b=10))
        viz_figures.append(overall_fig)
    
    # 2. Flagship Probability Gauge
    if 'premium_probability' in predictions:
        prob = predictions['premium_probability'] * 100
        flagship_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "XÁC SUẤT FLAGSHIP", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "green" if prob > 50 else "red"},
            }
        ))
        flagship_fig.update_layout(height=300, margin=dict(t=50, b=10))
        viz_figures.append(flagship_fig)
    
    # 3. Camera Rating
    if 'camera_rating' in predictions:
        rating = predictions['camera_rating']
        camera_fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=rating,
            number={'suffix': "/5", 'font': {'size': 40}},
            title={'text': "ĐÁNH GIÁ CAMERA", 'font': {'size': 16}},
            delta={'reference': 3}
        ))
        camera_fig.update_layout(height=300, margin=dict(t=50, b=10))
        viz_figures.append(camera_fig)
    
    return viz_figures

def create_gradio_interface():
    # Khởi tạo predictor
    try:
        predictor = MultiModelPredictor()
        print("✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        # Fallback: tạo predictor rỗng
        predictor = None
    
    with gr.Blocks(
        title="Hệ Thống Dự Đoán Điện Thoại",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("# 📱 Hệ Thống Dự Đoán Điện Thoại")
        gr.Markdown("Nhập thông số điện thoại để xem kết quả dự đoán với biểu đồ trực quan")
        
        with gr.Row():
            # Cột trái: Input - Nhập liệu chuyên sâu
            with gr.Column(scale=1):
                gr.Markdown("### ⌨️ Nhập Thông Số")
                
                # Service selection
                services = gr.CheckboxGroup(
                    choices=[
                        ("Đề xuất tổng quan", "recommender"),
                        ("Phát hiện flagship", "value_detector"), 
                        ("Đánh giá camera", "camera_predictor")
                    ],
                    label="Dịch vụ dự đoán",
                    value=["recommender", "value_detector", "camera_predictor"]
                )
                
                # Expert inputs với accordion
                with gr.Accordion("📱 Thông số màn hình", open=True):
                    with gr.Row():
                        screen_size = gr.Number(label="Kích thước màn hình (inch)", value=6.1)
                        ppi = gr.Number(label="Mật độ điểm ảnh (PPI)", value=460)
                    total_resolution = gr.Number(label="Độ phân giải tổng", value=2430000)
                
                with gr.Accordion("📸 Thông số camera", open=True):
                    with gr.Row():
                        camera_score = gr.Number(label="Điểm camera", value=65.0)
                        main_camera_mp = gr.Number(label="Camera chính (MP)", value=48.0)
                    with gr.Row():
                        num_cameras = gr.Number(label="Số lượng camera", value=3)
                        camera_feature_count = gr.Number(label="Số tính năng camera", value=2)
                    with gr.Row():
                        has_telephoto = gr.Checkbox(label="Có telephoto", value=True)
                        has_ultrawide = gr.Checkbox(label="Có ultrawide", value=True)
                        has_ois = gr.Checkbox(label="Có OIS", value=True)
                
                with gr.Accordion("⭐ Điểm đánh giá", open=False):
                    with gr.Row():
                        popularity_score = gr.Number(label="Điểm phổ biến", value=60.0)
                        overall_score_input = gr.Number(label="Điểm tổng quan", value=55.0)
                    with gr.Row():
                        display_score = gr.Number(label="Điểm màn hình", value=70.0)
                        camera_rating_input = gr.Number(label="Đánh giá camera", value=3.5)
                
                with gr.Accordion("💰 Thông số giá trị", open=False):
                    with gr.Row():
                        value_score = gr.Number(label="Điểm giá trị", value=6.5)
                        price_segment = gr.Radio(
                            choices=[("Phổ thông", 0), ("Tầm trung", 1), ("Cao cấp", 2)], 
                            label="Phân khúc giá",
                            value=1
                        )
                    is_premium_input = gr.Checkbox(label="Là flagship", value=False)
                
                with gr.Accordion("📦 Thông số sản phẩm", open=False):
                    with gr.Row():
                        has_warranty = gr.Checkbox(label="Có bảo hành", value=True)
                        number_of_review = gr.Number(label="Số đánh giá", value=120)
                
                predict_btn = gr.Button("🎯 Thực Hiện Dự Đoán", variant="primary", size="lg")
            
            # Cột phải: Kết quả + Biểu đồ
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Kết Quả Dự Đoán")
                
                # Kết quả dạng text
                with gr.Group():
                    gr.Markdown("#### Chi tiết kết quả")
                    overall_score_output = gr.Textbox(label="Điểm tổng quan", interactive=False)
                    flagship_output = gr.Textbox(label="Phân loại flagship", interactive=False)
                    camera_output = gr.Textbox(label="Đánh giá camera", interactive=False)
                    status_output = gr.Textbox(label="Trạng thái", value="Sẵn sàng", interactive=False)
                
                # Biểu đồ trực quan
                gr.Markdown("#### Biểu đồ trực quan")
                with gr.Row():
                    overall_viz = gr.Plot(label="Điểm tổng quan")
                    flagship_viz = gr.Plot(label="Xác suất flagship")
                with gr.Row():
                    camera_viz = gr.Plot(label="Đánh giá camera")
        
        # Hướng dẫn sử dụng
        gr.Markdown("---")
        gr.Markdown("### 💡 Hướng dẫn sử dụng")
        gr.Markdown("1. Chọn dịch vụ dự đoán cần sử dụng")
        gr.Markdown("2. Nhập các thông số điện thoại trong các mục tương ứng")  
        gr.Markdown("3. Nhấn 'Thực Hiện Dự Đoán' để xem kết quả và biểu đồ")
        gr.Markdown("4. Kết quả được dự đoán bằng Machine Learning models đã train")

        # ==================== EVENT HANDLERS ====================

        def handle_expert_prediction(services, screen_size, ppi, total_resolution,
                                   camera_score, main_camera_mp, num_cameras, camera_feature_count,
                                   has_telephoto, has_ultrawide, has_ois, popularity_score,
                                   overall_score_input, display_score, camera_rating_input,
                                   value_score, price_segment, is_premium_input,
                                   has_warranty, number_of_review):
            """Xử lý dự đoán từ manual input chuyên sâu"""
            
            if not services:
                return {
                    overall_score_output: "",
                    flagship_output: "", 
                    camera_output: "",
                    status_output: "❌ Vui lòng chọn ít nhất một dịch vụ",
                    overall_viz: None,
                    flagship_viz: None,
                    camera_viz: None
                }
            
            if predictor is None:
                return {
                    overall_score_output: "",
                    flagship_output: "",
                    camera_output: "",
                    status_output: "❌ Lỗi: Models chưa được load",
                    overall_viz: None,
                    flagship_viz: None,
                    camera_viz: None
                }
            
            # Chuẩn bị manual features
            manual_features = {
                "ScreenSize": screen_size,
                "PPI": ppi,
                "total_resolution": total_resolution,
                "camera_score": camera_score,
                "main_camera_mp": main_camera_mp,
                "num_cameras": num_cameras,
                "camera_feature_count": camera_feature_count,
                "has_telephoto": 1 if has_telephoto else 0,
                "has_ultrawide": 1 if has_ultrawide else 0,
                "has_ois": 1 if has_ois else 0,
                "popularity_score": popularity_score,
                "overall_score": overall_score_input,
                "display_score": display_score,
                "camera_rating": camera_rating_input,
                "value_score": value_score,
                "price_segment": price_segment,
                "is_premium": 1 if is_premium_input else 0,
                "has_warranty": 1 if has_warranty else 0,
                "NumberOfReview": number_of_review
            }
            
            try:
                # Gọi predictor
                result = predictor.predict_from_features(services, manual_features)
                
                if result['status'] == 'success':
                    predictions = result['predictions']
                    
                    # Format text outputs
                    overall_text = f"{predictions.get('overall_score', 'N/A')}/100" if 'overall_score' in predictions else "Chưa chọn dịch vụ"
                    
                    if 'is_premium' in predictions:
                        flagship_status = "📱 Flagship Phone" if predictions['is_premium'] else "📱 Phone thông thường"
                        prob = predictions.get('premium_probability', 0) * 100
                        flagship_text = f"{flagship_status} (Xác suất: {prob:.1f}%)"
                    else:
                        flagship_text = "Chưa chọn dịch vụ"
                        
                    camera_text = f"{predictions.get('camera_rating', 'N/A')}/5.0 ⭐" if 'camera_rating' in predictions else "Chưa chọn dịch vụ"
                    
                    # Create visualizations
                    viz_figures = create_visualizations(predictions)
                    
                    return {
                        overall_score_output: overall_text,
                        flagship_output: flagship_text,
                        camera_output: camera_text,
                        status_output: "✅ Dự đoán thành công!",
                        overall_viz: viz_figures[0] if len(viz_figures) > 0 else None,
                        flagship_viz: viz_figures[1] if len(viz_figures) > 1 else None,
                        camera_viz: viz_figures[2] if len(viz_figures) > 2 else None
                    }
                else:
                    return {
                        overall_score_output: "",
                        flagship_output: "",
                        camera_output: "",
                        status_output: f"❌ {result.get('error', 'Lỗi không xác định')}",
                        overall_viz: None,
                        flagship_viz: None,
                        camera_viz: None
                    }
                
            except Exception as e:
                return {
                    overall_score_output: "",
                    flagship_output: "",
                    camera_output: "",
                    status_output: f"❌ Lỗi dự đoán: {str(e)}",
                    overall_viz: None,
                    flagship_viz: None,
                    camera_viz: None
                }

        # Bind events
        predict_btn.click(
            handle_expert_prediction,
            inputs=[services, screen_size, ppi, total_resolution,
                   camera_score, main_camera_mp, num_cameras, camera_feature_count,
                   has_telephoto, has_ultrawide, has_ois, popularity_score,
                   overall_score_input, display_score, camera_rating_input,
                   value_score, price_segment, is_premium_input,
                   has_warranty, number_of_review],
            outputs=[overall_score_output, flagship_output, camera_output, status_output,
                    overall_viz, flagship_viz, camera_viz]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    print("✅ Gradio interface created successfully!")
    print("🤖 Using trained ML models for prediction")
    print("📊 Features: Manual input + Visualization charts") 
    print("🌐 Starting server on http://localhost:7869")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7855,
        share=False
    )