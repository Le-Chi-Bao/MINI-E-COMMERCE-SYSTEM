import gradio as gr
import requests
import os
import time
from typing import Dict, List, Tuple

API_URL = os.getenv("API_URL", "http://localhost:8000")

class PhonePricePredictorUI:
    def __init__(self):
        self.api_url = API_URL
    
    def predict_price(self, screen_size, resolution_width, resolution_height, 
                     main_camera_mp, num_cameras, has_telephoto, has_ultrawide, 
                     has_ois, has_warranty, number_of_reviews, model_name) -> Tuple:
        try:
            # Validate inputs
            screen_size = float(screen_size) if screen_size else 6.1
            resolution_width = int(resolution_width) if resolution_width else 1080
            resolution_height = int(resolution_height) if resolution_height else 2400
            main_camera_mp = float(main_camera_mp) if main_camera_mp else 12
            num_cameras = int(num_cameras) if num_cameras else 2
            number_of_reviews = int(number_of_reviews) if number_of_reviews else 100
            
            # Đảm bảo resolution_height không quá lớn (fix lỗi 422)
            if resolution_height > 10000:
                resolution_height = 4320  # Max reasonable value
            
            # Chuẩn bị features
            features = {
                "screen_size": screen_size,
                "resolution_width": resolution_width,
                "resolution_height": resolution_height,
                "main_camera_mp": main_camera_mp,
                "num_cameras": num_cameras,
                "has_telephoto": bool(has_telephoto),
                "has_ultrawide": bool(has_ultrawide),
                "has_ois": bool(has_ois),
                "has_warranty": bool(has_warranty),
                "number_of_reviews": number_of_reviews
            }
            
            print(f"📤 Gửi request đến API: {features}")
            
            # Thử gọi API
            endpoints = [
                f"{self.api_url}/api/v1/predict",
                f"{self.api_url}/predict", 
                f"{self.api_url}/api/predict"
            ]
            
            response = None
            for endpoint in endpoints:
                try:
                    payload = {
                        "phone_features": features,
                        "model_name": model_name
                    }
                    response = requests.post(endpoint, json=payload, timeout=30)
                    if response.status_code == 200:
                        print(f"✅ API response từ {endpoint}")
                        break
                    else:
                        print(f"❌ {endpoint}: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"❌ {endpoint} failed: {e}")
                    continue
            
            if response and response.status_code == 200:
                result = response.json()
                print(f"📥 API result: {result}")
                
                # Xử lý response
                predicted_price = result.get('predicted_price') or result.get('price') or 0
                model_used = result.get('model_used') or result.get('model') or model_name
                processing_time = result.get('processing_time') or result.get('time') or 0.1
                product_id = result.get('product_id') or f"PHONE_{int(time.time())}"
                
                # TRẢ VỀ 5 GIÁ TRỊ RIÊNG BIỆT (không phải dictionary)
                return (
                    f"<div class='price-result'>{float(predicted_price):,.0f} VND</div>",
                    model_used,
                    "85%",
                    f"{float(processing_time):.2f}s", 
                    product_id
                )
                
            else:
                # Fallback calculation
                print("⚠️ Using fallback calculation")
                base_price = 5000000  # 5 triệu
                price_multiplier = (
                    (screen_size / 6.1) * 
                    (main_camera_mp / 12) * 
                    (num_cameras / 2) *
                    (1.2 if has_telephoto else 1) *
                    (1.1 if has_ultrawide else 1) *
                    (1.1 if has_ois else 1) *
                    (1.05 if has_warranty else 1)
                )
                estimated_price = base_price * price_multiplier
                
                # TRẢ VỀ 5 GIÁ TRỊ RIÊNG BIỆT
                return (
                    f"<div class='price-result'>{estimated_price:,.0f} VND</div>",
                    f"{model_name} (local fallback)",
                    "65% (ước tính)",
                    "0.1s",
                    f"LOCAL_{int(time.time())}"
                )
                
        except Exception as e:
            error_msg = f"Lỗi: {str(e)}"
            print(f"❌ {error_msg}")
            
            # TRẢ VỀ 5 GIÁ TRỊ RIÊNG BIỆT CHO LỖI
            return (
                f"<div class='error-box'>{error_msg}</div>",
                "",
                "",
                "",
                ""
            )
    
    def get_available_models(self) -> List[str]:
        return ["kneighbors", "xgboost", "decisiontree", "linearregression"]

def create_interface():
    predictor_ui = PhonePricePredictorUI()
    
    with gr.Blocks(
        title="Phone Price Predictor", 
        theme=gr.themes.Soft(), 
        css="""
        .price-result {
            font-size: 2.5em;
            font-weight: bold;
            color: #22c55e;
            text-align: center;
            padding: 20px;
            border: 2px solid #22c55e;
            border-radius: 10px;
            background: #f0fdf4;
            margin: 10px 0;
        }
        .error-box {
            font-size: 1.2em;
            color: #ef4444;
            text-align: center;
            padding: 15px;
            border: 2px solid #ef4444;
            border-radius: 10px;
            background: #fef2f2;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 📱 Phone Price Predictor
        **Dự đoán giá điện thoại thông minh bằng AI**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🖥️ Thông số màn hình")
                screen_size = gr.Slider(4.0, 8.0, value=6.1, step=0.1, label="Kích thước màn hình (inch)")
                
                with gr.Row():
                    resolution_width = gr.Number(1170, label="Độ phân giải ngang (px)", precision=0, maximum=3840)
                    resolution_height = gr.Number(2532, label="Độ phân giải dọc (px)", precision=0, maximum=4320)
                
                gr.Markdown("### 📷 Thông số camera")
                main_camera_mp = gr.Slider(5, 200, value=12, step=1, label="Độ phân giải camera chính (MP)")
                num_cameras = gr.Slider(1, 5, value=3, step=1, label="Số lượng camera")
                
                with gr.Row():
                    has_telephoto = gr.Checkbox(label="📸 Camera Tele", value=True)
                    has_ultrawide = gr.Checkbox(label="🌅 Camera Siêu Rộng", value=True)
                    has_ois = gr.Checkbox(label="🔧 Chống rung quang học", value=True)
                
                gr.Markdown("### ℹ️ Thông tin sản phẩm")
                has_warranty = gr.Checkbox(label="📋 Có bảo hành", value=True)
                number_of_reviews = gr.Number(100, label="Số lượng đánh giá", precision=0, maximum=10000)
                
                gr.Markdown("### 🤖 Mô hình AI")
                model_name = gr.Dropdown(
                    choices=predictor_ui.get_available_models(),
                    value="kneighbors",
                    label="Chọn mô hình dự đoán"
                )
                
                predict_btn = gr.Button("🎯 Dự đoán giá", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 💰 Kết quả dự đoán")
                result_output = gr.HTML(value="<div class='price-result'>Nhập thông số và nhấn 'Dự đoán giá'</div>")
                
                with gr.Group():
                    gr.Markdown("**📊 Chi tiết kết quả:**")
                    model_used = gr.Textbox(label="Mô hình sử dụng", interactive=False)
                    confidence = gr.Textbox(label="Độ tin cậy", interactive=False)
                    processing_time = gr.Textbox(label="Thời gian xử lý", interactive=False)
                    product_id = gr.Textbox(label="Mã sản phẩm", interactive=False)
        
        # Examples với giá trị hợp lý
        examples = [
            [6.1, 1170, 2532, 12.0, 3, True, True, True, True, 200, "kneighbors"],
            [6.7, 1440, 3200, 48.0, 4, True, True, True, True, 500, "kneighbors"],
            [6.5, 1080, 2400, 50.0, 3, False, True, False, True, 80, "kneighbors"],
        ]
        
        gr.Examples(
            examples=examples, 
            inputs=[
                screen_size, resolution_width, resolution_height, main_camera_mp,
                num_cameras, has_telephoto, has_ultrawide, has_ois, 
                has_warranty, number_of_reviews, model_name
            ],
            label="📋 Ví dụ mẫu"
        )
        
        # XÓA FUNCTION update_result (không cần nữa)
        # Vì predict_price bây giờ trả về trực tiếp 5 giá trị
        
        predict_btn.click(
            fn=predictor_ui.predict_price,
            inputs=[
                screen_size, resolution_width, resolution_height, main_camera_mp,
                num_cameras, has_telephoto, has_ultrawide, has_ois, has_warranty,
                number_of_reviews, model_name
            ],
            outputs=[result_output, model_used, confidence, processing_time, product_id]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Khởi động Phone Price Predictor UI...")
    print(f"🌐 API URL: {API_URL}")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )