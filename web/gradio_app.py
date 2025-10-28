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
                    "screen_size": screen_size,
                    "resolution_width": resolution_width,
                    "resolution_height": resolution_height,
                    "main_camera_mp": main_camera_mp,
                    "num_cameras": num_cameras,
                    "has_telephoto": has_telephoto,
                    "has_ultrawide": has_ultrawide,
                    "has_ois": has_ois,
                    "has_warranty": has_warranty,
                    "number_of_reviews": number_of_reviews
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
                    "confidence": f"{result.get('confidence_score', 0) * 100:.1f}%",
                    "processing_time": f"{result['processing_time']:.2f}s",
                    "product_id": result['product_id']
                }
            else:
                return {"status": "error", "message": f"API Error: {response.text}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Request failed: {str(e)}"}
    
    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_url}/api/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [model['model_name'] for model in models]
            return ["xgboost", "decisiontree", "linearregression", "kneighbors"]
        except:
            return ["xgboost", "decisiontree", "linearregression", "kneighbors"]

def create_interface():
    predictor_ui = PhonePricePredictorUI()
    
    with gr.Blocks(title="Phone Price Predictor", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📱 Phone Price Predictor")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🖥️ Thông số màn hình")
                screen_size = gr.Slider(4.0, 8.0, value=6.7, step=0.1, label="Kích thước màn hình (inch)")
                
                with gr.Row():
                    resolution_width = gr.Number(2400, label="Độ phân giải ngang (px)")
                    resolution_height = gr.Number(1080, label="Độ phân giải dọc (px)")
                
                gr.Markdown("### 📷 Thông số camera")
                main_camera_mp = gr.Slider(5, 200, value=48, step=1, label="Độ phân giải camera chính (MP)")
                num_cameras = gr.Slider(1, 5, value=3, step=1, label="Số lượng camera")
                
                with gr.Row():
                    has_telephoto = gr.Checkbox(label="Camera Tele 📸")
                    has_ultrawide = gr.Checkbox(label="Camera Siêu Rộng 🌅")
                    has_ois = gr.Checkbox(label="Chống rung quang học 🔧")
                
                gr.Markdown("### ℹ️ Thông tin sản phẩm")
                has_warranty = gr.Checkbox(label="Có bảo hành 📋")
                number_of_reviews = gr.Number(100, label="Số lượng đánh giá")
                
                gr.Markdown("### 🤖 Mô hình AI")
                model_name = gr.Dropdown(
                    choices=predictor_ui.get_available_models(),
                    value="xgboost",
                    label="Chọn mô hình dự đoán"
                )
                
                predict_btn = gr.Button("🎯 Dự đoán giá", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 💰 Kết quả dự đoán")
                result_output = gr.HTML(value="<div class='price-result'>Nhập thông số và nhấn 'Dự đoán giá'</div>")
                
                with gr.Group():
                    gr.Markdown("**Chi tiết kết quả:**")
                    model_used = gr.Textbox(label="Mô hình sử dụng", interactive=False)
                    confidence = gr.Textbox(label="Độ tin cậy", interactive=False)
                    processing_time = gr.Textbox(label="Thời gian xử lý", interactive=False)
                    product_id = gr.Textbox(label="Mã sản phẩm", interactive=False)
        
        examples = [
            [6.1, 1170, 2532, 12.0, 2, False, False, False, True, 150, "xgboost"],
            [6.7, 2796, 1290, 48.0, 4, True, True, True, True, 500, "xgboost"],
            [6.5, 1080, 2400, 50.0, 3, False, True, False, True, 80, "xgboost"],
        ]
        
        gr.Examples(examples=examples, inputs=[screen_size, resolution_width, resolution_height, main_camera_mp,
                   num_cameras, has_telephoto, has_ultrawide, has_ois, has_warranty, number_of_reviews, model_name])
        
        def update_result(result):
            if result["status"] == "success":
                return (f"<div class='price-result'>{result['predicted_price']}</div>",
                        result["model_used"], result["confidence"], result["processing_time"], result["product_id"])
            else:
                return (f"<div class='error-box'>{result['message']}</div>", "", "", "", "")
        
        predict_btn.click(
            fn=predictor_ui.predict_price,
            inputs=[screen_size, resolution_width, resolution_height, main_camera_mp,
                   num_cameras, has_telephoto, has_ultrawide, has_ois, has_warranty,
                   number_of_reviews, model_name],
            outputs=[result_output, model_used, confidence, processing_time, product_id]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)