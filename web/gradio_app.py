import gradio as gr
import requests
import json
import pandas as pd
from typing import Dict, List

# API configuration
API_URL = "http://api:8000"

class PhonePredictionApp:
    def __init__(self):
        self.api_url = API_URL
        
    def get_services_info(self):
        """Lấy thông tin về các dịch vụ từ API"""
        try:
            response = requests.get(f"{self.api_url}/services")
            if response.status_code == 200:
                return response.json()['services']
            return {}
        except:
            return {}
    
    def predict_single_service(self, service: str, product_id: str):
        """Dự đoán cho một service"""
        try:
            response = requests.get(f"{self.api_url}/predict/{service}/{product_id}")
            if response.status_code == 200:
                return response.json()
            return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def predict_flexible(self, services: List[str], input_method: str, product_id: str, manual_features: Dict):
        """Dự đoán linh hoạt với các service được chọn"""
        try:
            payload = {
                "services": services,
                "input_method": input_method
            }
            
            if input_method == "feature_store":
                payload["product_id"] = product_id
            else:
                payload["manual_features"] = manual_features
            
            response = requests.post(f"{self.api_url}/predict", json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return {"error": f"API error: {error_detail}"}
                
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

def format_predictions(result):
    """Định dạng kết quả dự đoán"""
    if "error" in result:
        return f"❌ Lỗi: {result['error']}"
    
    predictions = result.get('predictions', {})
    
    output = "## 📊 Kết Quả Dự Đoán\n\n"
    
    if 'overall_score' in predictions:
        output += f"**🤖 Điểm Tổng Quan:** {predictions['overall_score']}/100\n"
        score = predictions['overall_score']
        if score >= 80:
            output += "   ⭐⭐⭐⭐⭐ - Xuất sắc!\n"
        elif score >= 60:
            output += "   ⭐⭐⭐⭐ - Tốt\n"
        elif score >= 40:
            output += "   ⭐⭐⭐ - Trung bình\n"
        else:
            output += "   ⭐⭐ - Cần cải thiện\n"
        output += "\n"
    
    if 'is_premium' in predictions:
        premium_status = "Có ✅" if predictions['is_premium'] else "Không ❌"
        prob = predictions.get('premium_probability', 0)
        output += f"**💰 Flagship Phone:** {premium_status}\n"
        output += f"   Xác suất: {prob:.1%}\n\n"
    
    if 'camera_rating' in predictions:
        rating = predictions['camera_rating']
        output += f"**📸 Đánh Giá Camera:** {rating}/5.0\n"
        stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
        output += f"   {stars}\n\n"
    
    # Thêm thông tin input method
    input_method = result.get('input_method', 'unknown')
    output += f"*Phương thức nhập: {input_method}*"
    
    return output

def create_gradio_interface():
    app = PhonePredictionApp()
    
    with gr.Blocks(
        title="Phone Prediction System",
        theme=gr.themes.Soft(),
        css="""
        .success-box { border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background: #f8fff8; }
        .error-box { border: 2px solid #f44336; padding: 10px; border-radius: 5px; background: #fff8f8; }
        .prediction-result { font-size: 16px; line-height: 1.6; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 📱 Hệ Thống Dự Đoán Điện Thoại
            **Dự đoán thông minh cho điện thoại sử dụng Machine Learning**
            """
        )
        
        # Tab 1: Dự đoán nhanh
        with gr.Tab("🚀 Dự Đoán Nhanh"):
            gr.Markdown("### Dự đoán nhanh theo Product ID")
            
            with gr.Row():
                with gr.Column():
                    quick_service = gr.Radio(
                        choices=["recommender", "value_detector", "camera_predictor", "all"],
                        label="Chọn Dịch Vụ Dự Đoán",
                        value="recommender",
                        info="Chọn dịch vụ bạn muốn sử dụng"
                    )
                    quick_product_id = gr.Textbox(
                        label="Product ID",
                        value="001",
                        placeholder="Nhập Product ID (ví dụ: 001, 050, 100)..."
                    )
                    quick_predict_btn = gr.Button("🎯 Dự Đoán Nhanh", variant="primary")
                
                with gr.Column():
                    quick_output = gr.Markdown()
        
        # Tab 2: Dự đoán linh hoạt
        with gr.Tab("🎛️ Dự Đoán Linh Hoạt"):
            gr.Markdown("### Dự đoán linh hoạt với nhiều dịch vụ")
            
            with gr.Row():
                with gr.Column():
                    # Service selection
                    services = gr.CheckboxGroup(
                        choices=[
                            ("🤖 Smart Recommender", "recommender"),
                            ("💰 Value Detector", "value_detector"), 
                            ("📸 Camera Predictor", "camera_predictor")
                        ],
                        label="Chọn Dịch Vụ",
                        value=["recommender"],
                        info="Chọn một hoặc nhiều dịch vụ"
                    )
                    
                    # Input method
                    input_method = gr.Radio(
                        choices=[
                            ("📁 Feature Store", "feature_store"),
                            ("⌨️ Manual Input", "manual")
                        ],
                        label="Phương Thức Nhập Liệu",
                        value="feature_store"
                    )
                    
                    # Product ID input (visible when feature_store selected)
                    product_id = gr.Textbox(
                        label="Product ID",
                        value="001",
                        visible=True,
                        placeholder="Nhập Product ID..."
                    )
                    
                    # Manual inputs container (visible when manual selected)
                    with gr.Column(visible=False) as manual_inputs_container:
                        gr.Markdown("### 📝 Nhập Thông Số Thủ Công")
                        
                        with gr.Accordion("📊 Display Features", open=True):
                            flex_screen_size = gr.Number(label="Screen Size (inches)", value=6.1)
                            flex_ppi = gr.Number(label="PPI (Pixels Per Inch)", value=460)
                            flex_total_resolution = gr.Number(label="Total Resolution", value=2430000)
                        
                        with gr.Accordion("📸 Camera Features", open=False):
                            flex_camera_score = gr.Number(label="Camera Score", value=65.0)
                            flex_main_camera_mp = gr.Number(label="Main Camera (MP)", value=48.0)
                            flex_num_cameras = gr.Number(label="Number of Cameras", value=3)
                            flex_has_telephoto = gr.Radio(choices=[0, 1], label="Has Telephoto", value=1)
                            flex_has_ultrawide = gr.Radio(choices=[0, 1], label="Has Ultrawide", value=1)
                            flex_has_ois = gr.Radio(choices=[0, 1], label="Has OIS", value=1)
                            flex_camera_feature_count = gr.Number(label="Camera Feature Count", value=2)
                        
                        with gr.Accordion("⭐ Rating Features", open=False):
                            flex_popularity_score = gr.Number(label="Popularity Score", value=60.0)
                            flex_overall_score = gr.Number(label="Overall Score", value=55.0)
                            flex_display_score = gr.Number(label="Display Score", value=70.0)
                            flex_camera_rating = gr.Number(label="Camera Rating", value=3.5)
                        
                        with gr.Accordion("💰 Value Features", open=False):
                            flex_value_score = gr.Number(label="Value Score", value=6.5)
                            flex_price_segment = gr.Radio(choices=[0, 1, 2], label="Price Segment (0=Budget, 1=Mid, 2=Premium)", value=1)
                            flex_is_premium = gr.Radio(choices=[0, 1], label="Is Premium", value=0)
                        
                        with gr.Accordion("📦 Product Features", open=False):
                            flex_has_warranty = gr.Radio(choices=[0, 1], label="Has Warranty", value=1)
                            flex_number_of_review = gr.Number(label="Number of Reviews", value=120)
            
                    flexible_predict_btn = gr.Button("🎯 Thực Hiện Dự Đoán", variant="primary")
                
                with gr.Column():
                    flexible_output = gr.Markdown()
        
        # Tab 3: Manual Input (chuyên sâu)
        with gr.Tab("⌨️ Nhập Liệu Thủ Công"):
            gr.Markdown("### Nhập thông số điện thoại thủ công")
            gr.Markdown("Điền các thông số bên dưới để dự đoán (chỉ cần nhập các features cần thiết cho dịch vụ đã chọn)")
            
            with gr.Row():
                with gr.Column():
                    manual_services = gr.CheckboxGroup(
                        choices=[
                            ("🤖 Smart Recommender", "recommender"),
                            ("💰 Value Detector", "value_detector"), 
                            ("📸 Camera Predictor", "camera_predictor")
                        ],
                        label="Chọn Dịch Vụ",
                        value=["recommender"]
                    )
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("📊 Display Features", open=True):
                        manual_screen_size = gr.Number(label="Screen Size (inches)", value=6.1)
                        manual_ppi = gr.Number(label="PPI (Pixels Per Inch)", value=460)
                        manual_total_resolution = gr.Number(label="Total Resolution", value=2430000)
                    
                    with gr.Accordion("📸 Camera Features", open=False):
                        manual_camera_score = gr.Number(label="Camera Score", value=65.0)
                        manual_main_camera_mp = gr.Number(label="Main Camera (MP)", value=48.0)
                        manual_num_cameras = gr.Number(label="Number of Cameras", value=3)
                        manual_has_telephoto = gr.Radio(choices=[0, 1], label="Has Telephoto", value=1)
                        manual_has_ultrawide = gr.Radio(choices=[0, 1], label="Has Ultrawide", value=1)
                        manual_has_ois = gr.Radio(choices=[0, 1], label="Has OIS", value=1)
                        manual_camera_feature_count = gr.Number(label="Camera Feature Count", value=2)
                
                with gr.Column():
                    with gr.Accordion("⭐ Rating Features", open=False):
                        manual_popularity_score = gr.Number(label="Popularity Score", value=60.0)
                        manual_overall_score = gr.Number(label="Overall Score", value=55.0)
                        manual_display_score = gr.Number(label="Display Score", value=70.0)
                        manual_camera_rating = gr.Number(label="Camera Rating", value=3.5)
                    
                    with gr.Accordion("💰 Value Features", open=False):
                        manual_value_score = gr.Number(label="Value Score", value=6.5)
                        manual_price_segment = gr.Radio(choices=[0, 1, 2], label="Price Segment (0=Budget, 1=Mid, 2=Premium)", value=1)
                        manual_is_premium = gr.Radio(choices=[0, 1], label="Is Premium", value=0)
                    
                    with gr.Accordion("📦 Product Features", open=False):
                        manual_has_warranty = gr.Radio(choices=[0, 1], label="Has Warranty", value=1)
                        manual_number_of_review = gr.Number(label="Number of Reviews", value=120)
            
            with gr.Row():
                manual_predict_btn = gr.Button("🎯 Dự Đoán Từ Manual Input", variant="primary", size="lg")
            
            manual_output = gr.Markdown()
        
        # Tab 4: Thông tin hệ thống
        with gr.Tab("ℹ️ Thông Tin Hệ Thống"):
            gr.Markdown("### Thông tin về các dịch vụ dự đoán")
            
            services_info = app.get_services_info()
            
            if services_info:
                for service_name, info in services_info.items():
                    with gr.Accordion(f"🔧 {service_name.upper()}", open=False):
                        gr.Markdown(f"**Đầu ra:** {info['output']}")
                        gr.Markdown(f"**Số features:** {info['feature_count']}")
                        gr.Markdown("**Features cần thiết:**")
                        
                        features_df = pd.DataFrame({
                            'Feature': info['required_features'],
                            'Type': ['Number' if any(c in f for c in ['Score', 'Size', 'PPI', 'mp', 'resolution']) else 
                                    'Binary' if 'has_' in f else 
                                    'Category' for f in info['required_features']]
                        })
                        
                        gr.Dataframe(features_df)
            else:
                gr.Markdown("❌ Không thể kết nối đến API. Vui lòng kiểm tra server.")
            
            gr.Markdown("---")
            gr.Markdown("### 📊 API Status")
            api_status = gr.HTML()
            
            def check_api_status():
                try:
                    response = requests.get(f"{API_URL}/health")
                    if response.status_code == 200:
                        data = response.json()
                        status_html = f"""
                        <div class="success-box">
                            <h3>✅ API Đang Hoạt Động</h3>
                            <p><strong>Status:</strong> {data['status']}</p>
                            <p><strong>Models loaded:</strong> {data['models_loaded']}</p>
                            <p><strong>Services:</strong> {', '.join(data['available_services'])}</p>
                        </div>
                        """
                    else:
                        status_html = f"<div class='error-box'><h3>❌ API Lỗi: {response.status_code}</h3></div>"
                except:
                    status_html = "<div class='error-box'><h3>❌ Không thể kết nối đến API</h3><p>Vui lòng kiểm tra server tại http://localhost:8000</p></div>"
                
                return status_html
            
            # Khởi tạo API status khi load page
            demo.load(check_api_status, outputs=api_status)
            gr.Button("🔄 Kiểm Tra Lại").click(check_api_status, outputs=api_status)
        
        # Event handlers
        def handle_quick_prediction(service, product_id):
            """Xử lý dự đoán nhanh"""
            result = app.predict_single_service(service, product_id)
            return format_predictions(result)
        
        def toggle_input_method(input_method):
            """Hiển thị input phù hợp với phương thức được chọn"""
            if input_method == "feature_store":
                return [
                    gr.Textbox(visible=True),  # product_id
                    gr.Column(visible=False)   # manual_inputs_container
                ]
            else:
                return [
                    gr.Textbox(visible=False), # product_id  
                    gr.Column(visible=True)    # manual_inputs_container
                ]
        
        def handle_flexible_prediction(services, input_method, product_id, 
                                     flex_screen_size, flex_ppi, flex_total_resolution,
                                     flex_camera_score, flex_main_camera_mp, flex_num_cameras,
                                     flex_has_telephoto, flex_has_ultrawide, flex_has_ois,
                                     flex_camera_feature_count, flex_popularity_score,
                                     flex_overall_score, flex_display_score, flex_camera_rating,
                                     flex_value_score, flex_price_segment, flex_is_premium,
                                     flex_has_warranty, flex_number_of_review):
            """Xử lý dự đoán linh hoạt với cả 2 phương thức input"""
            if input_method == "feature_store":
                result = app.predict_flexible(services, input_method, product_id, {})
            else:
                # Manual input - thu thập tất cả features
                manual_features = {
                    "ScreenSize": flex_screen_size,
                    "PPI": flex_ppi,
                    "total_resolution": flex_total_resolution,
                    "camera_score": flex_camera_score,
                    "main_camera_mp": flex_main_camera_mp,
                    "num_cameras": flex_num_cameras,
                    "has_telephoto": flex_has_telephoto,
                    "has_ultrawide": flex_has_ultrawide,
                    "has_ois": flex_has_ois,
                    "camera_feature_count": flex_camera_feature_count,
                    "popularity_score": flex_popularity_score,
                    "overall_score": flex_overall_score,
                    "display_score": flex_display_score,
                    "camera_rating": flex_camera_rating,
                    "value_score": flex_value_score,
                    "price_segment": flex_price_segment,
                    "is_premium": flex_is_premium,
                    "has_warranty": flex_has_warranty,
                    "NumberOfReview": flex_number_of_review
                }
                # Loại bỏ các giá trị None
                manual_features = {k: v for k, v in manual_features.items() if v is not None}
                result = app.predict_flexible(services, input_method, "", manual_features)
            
            return format_predictions(result)
        
        def handle_manual_prediction(services, 
                                   manual_screen_size, manual_ppi, manual_total_resolution,
                                   manual_camera_score, manual_main_camera_mp, manual_num_cameras,
                                   manual_has_telephoto, manual_has_ultrawide, manual_has_ois,
                                   manual_camera_feature_count, manual_popularity_score,
                                   manual_overall_score, manual_display_score, manual_camera_rating,
                                   manual_value_score, manual_price_segment, manual_is_premium,
                                   manual_has_warranty, manual_number_of_review):
            """Xử lý dự đoán từ manual input chuyên sâu"""
            # Thu thập tất cả features
            manual_features = {
                "ScreenSize": manual_screen_size,
                "PPI": manual_ppi,
                "total_resolution": manual_total_resolution,
                "camera_score": manual_camera_score,
                "main_camera_mp": manual_main_camera_mp,
                "num_cameras": manual_num_cameras,
                "has_telephoto": manual_has_telephoto,
                "has_ultrawide": manual_has_ultrawide,
                "has_ois": manual_has_ois,
                "camera_feature_count": manual_camera_feature_count,
                "popularity_score": manual_popularity_score,
                "overall_score": manual_overall_score,
                "display_score": manual_display_score,
                "camera_rating": manual_camera_rating,
                "value_score": manual_value_score,
                "price_segment": manual_price_segment,
                "is_premium": manual_is_premium,
                "has_warranty": manual_has_warranty,
                "NumberOfReview": manual_number_of_review
            }
            # Loại bỏ các giá trị None
            manual_features = {k: v for k, v in manual_features.items() if v is not None}
            result = app.predict_flexible(services, "manual", "", manual_features)
            return format_predictions(result)
        
        # Bind events
        
        # Tab 1: Dự đoán nhanh
        quick_predict_btn.click(
            handle_quick_prediction,
            inputs=[quick_service, quick_product_id],
            outputs=quick_output
        )
        
        # Tab 2: Dự đoán linh hoạt
        input_method.change(
            toggle_input_method,
            inputs=input_method,
            outputs=[product_id, manual_inputs_container]
        )
        
        flexible_predict_btn.click(
            handle_flexible_prediction,
            inputs=[
                services, input_method, product_id,
                flex_screen_size, flex_ppi, flex_total_resolution,
                flex_camera_score, flex_main_camera_mp, flex_num_cameras,
                flex_has_telephoto, flex_has_ultrawide, flex_has_ois,
                flex_camera_feature_count, flex_popularity_score,
                flex_overall_score, flex_display_score, flex_camera_rating,
                flex_value_score, flex_price_segment, flex_is_premium,
                flex_has_warranty, flex_number_of_review
            ],
            outputs=flexible_output
        )
        
        # Tab 3: Manual input chuyên sâu
        manual_predict_btn.click(
            handle_manual_prediction,
            inputs=[
                manual_services,
                manual_screen_size, manual_ppi, manual_total_resolution,
                manual_camera_score, manual_main_camera_mp, manual_num_cameras,
                manual_has_telephoto, manual_has_ultrawide, manual_has_ois,
                manual_camera_feature_count, manual_popularity_score,
                manual_overall_score, manual_display_score, manual_camera_rating,
                manual_value_score, manual_price_segment, manual_is_premium,
                manual_has_warranty, manual_number_of_review
            ],
            outputs=manual_output
        )
        
        gr.Markdown("---")
        gr.Markdown(
            """
            ### 💡 Hướng Dẫn Sử Dụng:
            
            #### 🚀 Dự Đoán Nhanh:
            - Chọn một dịch vụ và nhập Product ID
            - Product ID hợp lệ: 001, 002, ..., 100 (từ dữ liệu training)
            
            #### 🎛️ Dự Đoán Linh Hoạt:
            - **Feature Store**: Chọn nhiều dịch vụ + nhập Product ID
            - **Manual Input**: Chọn nhiều dịch vụ + nhập thông số thủ công
            
            #### ⌨️ Nhập Liệu Thủ Công:
            - Form chuyên sâu để nhập tất cả thông số
            - Phù hợp khi không có Product ID
            
            🚀 *Hệ thống sử dụng Machine Learning để dự đoán với độ chính xác cao*
            """
        )
    
    return demo

if __name__ == "__main__":
    # Kiểm tra phiên bản Gradio
    import gradio as gr
    print(f"🚀 Gradio version: {gr.__version__}")
    
    demo = create_gradio_interface()
    print("✅ Gradio interface created successfully!")
    print("🌐 Starting server on http://localhost:7869")
    print("📱 Available tabs:")
    print("   - 🚀 Dự Đoán Nhanh")
    print("   - 🎛️ Dự Đoán Linh Hoạt") 
    print("   - ⌨️ Nhập Liệu Thủ Công")
    print("   - ℹ️ Thông Tin Hệ Thống")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7876,
        share=False,
        show_error=True
    )