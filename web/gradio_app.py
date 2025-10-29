import gradio as gr
import requests
import os
from typing import Dict, List

API_URL = os.getenv("API_URL", "http://localhost:8000")

class PhonePricePredictorUI:
    def __init__(self):
        self.api_url = API_URL
    
    def predict_price(self, screen_size, resolution_width, resolution_height, 
                     main_camera_mp, num_cameras, has_telephoto, has_ultrawide, 
                     has_ois, has_warranty, number_of_reviews, model_name) -> Dict:
        try:
            features = {
                "phone_features": {
                    "screen_size": float(screen_size),
                    "resolution_width": int(resolution_width),
                    "resolution_height": int(resolution_height),
                    "main_camera_mp": float(main_camera_mp),
                    "num_cameras": int(num_cameras),
                    "has_telephoto": bool(has_telephoto),
                    "has_ultrawide": bool(has_ultrawide),
                    "has_ois": bool(has_ois),
                    "has_warranty": bool(has_warranty),
                    "number_of_reviews": float(number_of_reviews)
                },
                "model_name": model_name
            }
            
            response = requests.post(f"{self.api_url}/api/v1/predict", json=features, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "predicted_price": f"{result['predicted_price']:,.0f} VND",
                    "model_used": result['model_used'],
                    "confidence": f"{result.get('confidence_score', 0.85) * 100:.1f}%",  # ✅ DEFAULT VALUE
                    "processing_time": f"{result['processing_time']:.2f}s",
                    "product_id": result['product_id']
                }
            else:
                error_detail = response.text[:100] if response.text else "Unknown error"
                return {"status": "error", "message": f"API Error {response.status_code}: {error_detail}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Request failed: {str(e)}"}
    
    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_url}/api/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [model['model_name'] for model in models]
            return ["kneighbors", "xgboost", "decisiontree", "linearregression"]  # ✅ SẮP XẾP THEO PERFORMANCE
        except:
            return ["kneighbors", "xgboost", "decisiontree", "linearregression"]

def create_interface():
    predictor_ui = PhonePricePredictorUI()
    
    with gr.Blocks(title="Phone Price Predictor", theme=gr.themes.Soft(), css="""
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
    """) as interface:
        
        gr.Markdown("""
        # 📱 Phone Price Predictor
        **Dự đoán giá điện thoại thông minh bằng AI**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🖥️ Thông số màn hình")
                screen_size = gr.Slider(4.0, 8.0, value=6.1, step=0.1, label="Kích thước màn hình (inch)")
                
                with gr.Row():
                    resolution_width = gr.Number(1170, label="Độ phân giải ngang (px)", precision=0)
                    resolution_height = gr.Number(2532, label="Độ phân giải dọc (px)", precision=0)
                
                gr.Markdown("### 📷 Thông số camera")
                main_camera_mp = gr.Slider(5, 200, value=12, step=1, label="Độ phân giải camera chính (MP)")
                num_cameras = gr.Slider(1, 5, value=3, step=1, label="Số lượng camera")
                
                with gr.Row():
                    has_telephoto = gr.Checkbox(label="📸 Camera Tele", value=True)
                    has_ultrawide = gr.Checkbox(label="🌅 Camera Siêu Rộng", value=True)
                    has_ois = gr.Checkbox(label="🔧 Chống rung quang học", value=True)
                
                gr.Markdown("### ℹ️ Thông tin sản phẩm")
                has_warranty = gr.Checkbox(label="📋 Có bảo hành", value=True)
                number_of_reviews = gr.Number(100, label="Số lượng đánh giá", precision=0)
                
                gr.Markdown("### 🤖 Mô hình AI")
                model_name = gr.Dropdown(
                    choices=predictor_ui.get_available_models(),
                    value="kneighbors",  # ✅ DEFAULT LÀ MODEL TỐT NHẤT
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
        
        # ✅ SỬA EXAMPLES - ĐẢO NGƯỢC WIDTH/HEIGHT CHO ĐÚNG
        examples = [
            [6.1, 1170, 2532, 12.0, 3, True, True, True, True, 200, "kneighbors"],  # iPhone
            [6.7, 1290, 2796, 48.0, 4, True, True, True, True, 500, "kneighbors"],  # iPhone Pro
            [6.5, 1080, 2400, 50.0, 3, False, True, False, True, 80, "kneighbors"], # Mid-range
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
        
        def update_result(result):
            if result["status"] == "success":
                return (
                    f"<div class='price-result'>{result['predicted_price']}</div>",
                    result["model_used"], 
                    result["confidence"], 
                    result["processing_time"], 
                    result["product_id"]
                )
            else:
                return (
                    f"<div class='error-box'>{result['message']}</div>", 
                    "", "", "", ""
                )
        
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
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False  # ✅ TẮT SHARE PUBLIC (chỉ local)
    )