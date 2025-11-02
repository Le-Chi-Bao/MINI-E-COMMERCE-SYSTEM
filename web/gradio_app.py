# web/gradio_app.py
import gradio as gr
import requests
import pandas as pd
from typing import Dict, List
import json

API_URL = "http://api:8000"
print(f"🔗 Connecting to API at: {API_URL}")

class PhonePredictionApp:
    def __init__(self):
        self.api_url = API_URL
        self.current_user = None
        self.token = None
        
    def _safe_request(self, method, endpoint, json_data=None):
        """Wrapper for safe requests with proper encoding"""
        try:
            url = f"{self.api_url}{endpoint}"
            headers = {
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            }
            
            if self.token:
                headers['Authorization'] = f"Bearer {self.token}"
            
            if json_data:
                response = requests.request(
                    method, 
                    url, 
                    data=json.dumps(json_data, ensure_ascii=False).encode('utf-8'),
                    headers=headers,
                    timeout=30
                )
            else:
                response = requests.request(
                    method, 
                    url, 
                    headers=headers,
                    timeout=30
                )
                
            return response
        except Exception as e:
            print(f"Request error: {e}")
            raise
        
    def login(self, username: str, password: str) -> Dict:
        """Đăng nhập vào hệ thống"""
        try:
            response = self._safe_request(
                'POST', 
                '/auth/login',
                json_data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                return {"success": False, "message": f"Lỗi server: {response.status_code}"}
                
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": f"Không thể kết nối đến API tại {self.api_url}"}
        except Exception as e:
            return {"success": False, "message": f"Lỗi: {str(e)}"}
    
    def register(self, username: str, password: str, email: str = "") -> Dict:
        """Đăng ký tài khoản mới"""
        try:
            response = self._safe_request(
                'POST',
                '/auth/register', 
                json_data={"username": username, "password": password, "email": email}
            )
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": f"Không thể kết nối đến API tại {self.api_url}"}
        except Exception as e:
            return {"success": False, "message": f"Lỗi: {str(e)}"}
    
    def logout(self):
        """Đăng xuất"""
        self.current_user = None
        self.token = None
    
    def is_logged_in(self) -> bool:
        """Kiểm tra đã đăng nhập chưa"""
        return self.current_user is not None
    
    def get_services_info(self):
        """Lấy thông tin về các dịch vụ từ API"""
        try:
            response = self._safe_request('GET', '/services')
            if response.status_code == 200:
                return response.json()['services']
            return {}
        except:
            return {}
    
    def predict_single_service(self, service: str, product_id: str):
        """Dự đoán cho một service"""
        try:
            response = self._safe_request('GET', f'/predict/{service}/{product_id}')
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
            
            response = self._safe_request('POST', '/predict', json_data=payload)
            
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
        return f"""
        <div style="background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 8px;">
            <h3 style="color: #d00; margin: 0;">Lỗi hệ thống</h3>
            <p style="margin: 10px 0 0 0; color: #900;">{result['error']}</p>
        </div>
        """
    
    predictions = result.get('predictions', {})
    
    output = """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0;">
            <h2 style="margin: 0; text-align: center;">KẾT QUẢ DỰ ĐOÁN</h2>
        </div>
        <div style="background: white; padding: 25px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    """
    
    if 'overall_score' in predictions:
        score = predictions['overall_score']
        rating_class = ""
        if score >= 80:
            rating_class = "xuất-sắc"
            rating_text = "XUẤT SẮC"
        elif score >= 60:
            rating_class = "tốt"
            rating_text = "TỐT"
        elif score >= 40:
            rating_class = "trung-bình"
            rating_text = "TRUNG BÌNH"
        else:
            rating_class = "yếu"
            rating_text = "CẦN CẢI THIỆN"
            
        output += f"""
            <div class="metric-card">
                <div class="metric-header">
                    <h3 style="color: #2c3e50; margin: 0 0 15px 0;">Điểm Đánh Giá Tổng Quan</h3>
                </div>
                <div class="score-display {rating_class}">
                    <span class="score-value">{score}</span>
                    <span class="score-max">/100</span>
                </div>
                <div class="rating-badge {rating_class}">
                    {rating_text}
                </div>
            </div>
        """
    
    if 'is_premium' in predictions:
        is_premium = predictions['is_premium']
        prob = predictions.get('premium_probability', 0) * 100
        
        output += f"""
            <div class="metric-card">
                <div class="metric-header">
                    <h3 style="color: #2c3e50; margin: 0 0 15px 0;">Phân Loại Flagship</h3>
                </div>
                <div class="premium-status {'premium' if is_premium else 'standard'}">
                    <span class="status-indicator"></span>
                    <span class="status-text">{'FLAGSHIP PHONE' if is_premium else 'PHONE THÔNG THƯỜNG'}</span>
                </div>
                <div class="probability">
                    Xác suất: <strong>{prob:.1f}%</strong>
                </div>
            </div>
        """
    
    if 'camera_rating' in predictions:
        rating = predictions['camera_rating']
        stars_full = int(rating)
        stars_half = 1 if rating - stars_full >= 0.5 else 0
        stars_empty = 5 - stars_full - stars_half
        
        stars_html = "★" * stars_full + "☆" * stars_empty
        
        output += f"""
            <div class="metric-card">
                <div class="metric-header">
                    <h3 style="color: #2c3e50; margin: 0 0 15px 0;">Đánh Giá Hệ Thống Camera</h3>
                </div>
                <div class="camera-rating">
                    <div class="stars">{stars_html}</div>
                    <div class="rating-value">{rating}/5.0</div>
                </div>
            </div>
        """
    
    # Thêm thông tin user
    user = result.get('user', 'Unknown')
    output += f"""
            <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #ecf0f1; text-align: center;">
                <small style="color: #7f8c8d;">Người dùng: <strong>{user}</strong></small>
            </div>
        </div>
    </div>
    
    <style>
    .metric-card {{
        background: #f8f9fa;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }}
    
    .score-display {{
        text-align: center;
        margin: 15px 0;
    }}
    
    .score-value {{
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
    }}
    
    .score-max {{
        font-size: 1.2em;
        color: #7f8c8d;
    }}
    
    .rating-badge {{
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.9em;
    }}
    
    .xuất-sắc {{ background: #27ae60; color: white; }}
    .tốt {{ background: #2ecc71; color: white; }}
    .trung-bình {{ background: #f39c12; color: white; }}
    .yếu {{ background: #e74c3c; color: white; }}
    
    .premium-status {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 15px 0;
    }}
    
    .status-indicator {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
    }}
    
    .premium .status-indicator {{ background: #e74c3c; }}
    .standard .status-indicator {{ background: #27ae60; }}
    
    .status-text {{
        font-weight: bold;
        font-size: 1.1em;
    }}
    
    .premium .status-text {{ color: #e74c3c; }}
    .standard .status-text {{ color: #27ae60; }}
    
    .probability {{
        text-align: center;
        color: #7f8c8d;
        margin-top: 10px;
    }}
    
    .camera-rating {{
        text-align: center;
        margin: 15px 0;
    }}
    
    .stars {{
        font-size: 2em;
        color: #f39c12;
        margin-bottom: 10px;
    }}
    
    .rating-value {{
        font-size: 1.2em;
        color: #2c3e50;
        font-weight: bold;
    }}
    </style>
    """
    
    return output

def create_gradio_interface():
    app = PhonePredictionApp()
    
    with gr.Blocks(
        title="Hệ Thống Dự Đoán Điện Thoại",
        theme=gr.themes.Monochrome(
            primary_hue="blue",
            secondary_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
            spacing_size="lg",
            radius_size="lg"
        )
    ) as demo:
        
        # Header
        gr.Markdown("# Hệ Thống Dự Đoán Điện Thoại Thông Minh")
        gr.Markdown("Sử dụng Machine Learning để đánh giá và dự đoán hiệu năng điện thoại")
        
        # Biến state để lưu trạng thái đăng nhập
        current_user_state = gr.State(value=None)
        
        # Tab Đăng nhập/Đăng ký
        with gr.Tab("Xác Thực"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Đăng Nhập Hệ Thống")
                    with gr.Row():
                        login_username = gr.Textbox(
                            label="Tên đăng nhập",
                            placeholder="Nhập username...",
                            scale=2
                        )
                        login_password = gr.Textbox(
                            label="Mật khẩu", 
                            type="password", 
                            placeholder="Nhập mật khẩu...",
                            scale=2
                        )
                    login_btn = gr.Button("Đăng Nhập", variant="primary", size="lg")
                    login_status = gr.Markdown()
                    
                    gr.Markdown("---")
                    gr.Markdown("**Tài khoản demo:**")
                    gr.Markdown("- Username: `admin`")
                    gr.Markdown("- Password: `admin`")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Đăng Ký Tài Khoản Mới")
                    with gr.Row():
                        reg_username = gr.Textbox(label="Tên đăng nhập", placeholder="Chọn username...")
                        reg_password = gr.Textbox(label="Mật khẩu", type="password", placeholder="Tạo mật khẩu...")
                    reg_email = gr.Textbox(label="Email (tùy chọn)", placeholder="your@email.com")
                    register_btn = gr.Button("Đăng Ký", variant="secondary")
                    register_status = gr.Markdown()

        # Tab chính (chỉ hiện khi đã login)
        with gr.Tab("Dự Đoán", visible=False) as main_tab:
            
            # User info panel
            with gr.Row():
                with gr.Column(scale=8):
                    user_info_display = gr.Markdown()
                with gr.Column(scale=2):
                    logout_btn = gr.Button("Đăng Xuất", variant="stop", size="lg")
            
            # Phân chia rõ ràng 3 phương thức dự đoán
            with gr.Tabs():
                # TAB 1: Dự đoán nhanh (Feature Store only)
                with gr.TabItem("Dự Đoán Nhanh"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Cấu Hình Dự Đoán")
                            quick_service = gr.Radio(
                                choices=[
                                    ("Đề xuất tổng quan", "recommender"),
                                    ("Phát hiện flagship", "value_detector"), 
                                    ("Đánh giá camera", "camera_predictor"),
                                    ("Tất cả dịch vụ", "all")
                                ],
                                label="Dịch vụ dự đoán",
                                value="recommender"
                            )
                            quick_product_id = gr.Textbox(
                                label="Mã sản phẩm (Product ID)",
                                value="001",
                                placeholder="Nhập ID sản phẩm (001-100)...",
                                info="Sử dụng dữ liệu từ Feature Store"
                            )
                            gr.Markdown("**Mã sản phẩm hợp lệ:** 001-100")
                            quick_predict_btn = gr.Button("Thực Hiện Dự Đoán", variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Kết Quả Dự Đoán")
                            quick_output = gr.HTML()
                
                # TAB 2: Dự đoán linh hoạt (Feature Store + Basic Manual)
                with gr.TabItem("Dự Đoán Nâng Cao"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Cấu Hình Nâng Cao")
                            services = gr.CheckboxGroup(
                                choices=[
                                    ("Đề xuất tổng quan", "recommender"),
                                    ("Phát hiện flagship", "value_detector"), 
                                    ("Đánh giá camera", "camera_predictor")
                                ],
                                label="Dịch vụ được chọn",
                                value=["recommender"],
                                info="Chọn một hoặc nhiều dịch vụ"
                            )
                            
                            input_method = gr.Radio(
                                choices=[
                                    ("Sử dụng Feature Store", "feature_store"),
                                    ("Nhập thông số cơ bản", "manual")
                                ],
                                label="Phương thức nhập liệu",
                                value="feature_store"
                            )
                            
                            # Product ID input (chỉ hiện khi chọn Feature Store)
                            product_id = gr.Textbox(
                                label="Mã sản phẩm",
                                value="001",
                                visible=True,
                                placeholder="Nhập Product ID..."
                            )
                            
                            # Basic manual inputs (chỉ hiện khi chọn Manual)
                            with gr.Column(visible=False) as basic_manual_inputs:
                                gr.Markdown("### Thông Số Cơ Bản")
                                with gr.Row():
                                    basic_screen_size = gr.Number(label="Kích thước màn hình (inch)", value=6.1)
                                    basic_ppi = gr.Number(label="Mật độ điểm ảnh (PPI)", value=460)
                                with gr.Row():
                                    basic_camera_score = gr.Number(label="Điểm camera", value=65.0)
                                    basic_main_camera = gr.Number(label="Camera chính (MP)", value=48.0)
                                with gr.Row():
                                    basic_value_score = gr.Number(label="Điểm giá trị", value=6.5)
                                    basic_reviews = gr.Number(label="Số lượng đánh giá", value=120)
                            
                            advanced_predict_btn = gr.Button("Thực Hiện Dự Đoán", variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Kết Quả Dự Đoán")
                            advanced_output = gr.HTML()
                
                # TAB 3: Nhập liệu chuyên sâu (Comprehensive Manual Input)
                with gr.TabItem("Nhập Liệu Chuyên Sâu"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Cấu Hình Chuyên Sâu")
                            expert_services = gr.CheckboxGroup(
                                choices=[
                                    ("Đề xuất tổng quan", "recommender"),
                                    ("Phát hiện flagship", "value_detector"), 
                                    ("Đánh giá camera", "camera_predictor")
                                ],
                                label="Dịch vụ được chọn",
                                value=["recommender"]
                            )
                            
                            # Expert inputs với accordion
                            gr.Markdown("### Thông Số Chi Tiết")
                            
                            with gr.Accordion("Thông số màn hình", open=True):
                                with gr.Row():
                                    expert_screen_size = gr.Number(label="Kích thước màn hình (inch)", value=6.1)
                                    expert_ppi = gr.Number(label="Mật độ điểm ảnh (PPI)", value=460)
                                expert_total_resolution = gr.Number(label="Độ phân giải tổng", value=2430000)
                            
                            with gr.Accordion("Thông số camera", open=False):
                                with gr.Row():
                                    expert_camera_score = gr.Number(label="Điểm camera", value=65.0)
                                    expert_main_camera = gr.Number(label="Camera chính (MP)", value=48.0)
                                with gr.Row():
                                    expert_num_cameras = gr.Number(label="Số lượng camera", value=3)
                                    expert_camera_features = gr.Number(label="Số tính năng camera", value=2)
                                with gr.Row():
                                    expert_has_telephoto = gr.Radio(choices=[("Có", 1), ("Không", 0)], label="Telephoto", value=1)
                                    expert_has_ultrawide = gr.Radio(choices=[("Có", 1), ("Không", 0)], label="Ultrawide", value=1)
                                    expert_has_ois = gr.Radio(choices=[("Có", 1), ("Không", 0)], label="OIS", value=1)
                            
                            with gr.Accordion("Điểm đánh giá", open=False):
                                with gr.Row():
                                    expert_popularity = gr.Number(label="Điểm phổ biến", value=60.0)
                                    expert_overall = gr.Number(label="Điểm tổng quan", value=55.0)
                                with gr.Row():
                                    expert_display_score = gr.Number(label="Điểm màn hình", value=70.0)
                                    expert_camera_rating = gr.Number(label="Đánh giá camera", value=3.5)
                            
                            with gr.Accordion("Thông số giá trị", open=False):
                                with gr.Row():
                                    expert_value_score = gr.Number(label="Điểm giá trị", value=6.5)
                                    expert_price_segment = gr.Radio(
                                        choices=[("Phổ thông", 0), ("Tầm trung", 1), ("Cao cấp", 2)], 
                                        label="Phân khúc giá",
                                        value=1
                                    )
                                expert_is_premium = gr.Radio(choices=[("Có", 1), ("Không", 0)], label="Flagship", value=0)
                            
                            with gr.Accordion("Thông số sản phẩm", open=False):
                                with gr.Row():
                                    expert_has_warranty = gr.Radio(choices=[("Có", 1), ("Không", 0)], label="Bảo hành", value=1)
                                    expert_reviews = gr.Number(label="Số đánh giá", value=120)
                            
                            expert_predict_btn = gr.Button("Thực Hiện Dự Đoán", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Kết Quả Dự Đoán")
                            expert_output = gr.HTML()
                
                # TAB 4: Thông tin hệ thống
                with gr.TabItem("Thông Tin Hệ Thống"):
                    gr.Markdown("### Thông tin về các dịch vụ dự đoán")
                    
                    services_info = app.get_services_info()
                    
                    if services_info:
                        for service_name, info in services_info.items():
                            with gr.Accordion(f"Dịch vụ: {service_name.upper()}", open=False):
                                gr.Markdown(f"**Đầu ra:** {info['output']}")
                                gr.Markdown(f"**Số lượng features:** {info['feature_count']}")
                                gr.Markdown("**Features yêu cầu:**")
                                
                                features_df = pd.DataFrame({
                                    'Feature': info['required_features'],
                                    'Type': ['Số' if any(c in f for c in ['Score', 'Size', 'PPI', 'mp', 'resolution']) else 
                                            'Nhị phân' if 'has_' in f else 
                                            'Phân loại' for f in info['required_features']]
                                })
                                
                                gr.Dataframe(features_df)
                    else:
                        gr.Markdown("Không thể kết nối đến API server.")

        # Event handlers
        def handle_login(username, password):
            result = app.login(username, password)
            if result["success"]:
                user_info = result["user_info"]
                app.current_user = user_info
                app.token = username
                welcome_msg = f"""
                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <h3 style="margin: 0 0 5px 0; color: #2c3e50;">Chào mừng {user_info['username']}!</h3>
                    <p style="margin: 0; color: #27ae60;">Đăng nhập thành công</p>
                </div>
                """
                return {
                    current_user_state: user_info,
                    login_status: "",
                    main_tab: gr.update(visible=True),
                    user_info_display: welcome_msg
                }
            else:
                return {
                    current_user_state: None,
                    login_status: f"**Lỗi:** {result['message']}",
                    main_tab: gr.update(visible=False),
                    user_info_display: ""
                }
        
        def handle_register(username, password, email):
            result = app.register(username, password, email)
            status_msg = f"**Thành công:** {result['message']}" if result["success"] else f"**Lỗi:** {result['message']}"
            return {register_status: status_msg}
        
        def handle_logout():
            app.logout()
            return {
                current_user_state: None,
                main_tab: gr.update(visible=False),
                user_info_display: "",
                login_status: ""
            }
        
        def toggle_input_method(input_method):
            if input_method == "feature_store":
                return [gr.Textbox(visible=True), gr.Column(visible=False)]
            else:
                return [gr.Textbox(visible=False), gr.Column(visible=True)]
        
        def handle_quick_prediction(service, product_id, current_user):
            if not current_user:
                return "<div style='color: #e74c3c; padding: 20px; text-align: center;'>Vui lòng đăng nhập để sử dụng tính năng này</div>"
            result = app.predict_single_service(service, product_id)
            return format_predictions(result)
        
        def handle_advanced_prediction(services, input_method, product_id, current_user, 
                                     screen_size, ppi, camera_score, main_camera, value_score, reviews):
            if not current_user:
                return "<div style='color: #e74c3c; padding: 20px; text-align: center;'>Vui lòng đăng nhập để sử dụng tính năng này</div>"
            
            if input_method == "feature_store":
                result = app.predict_flexible(services, input_method, product_id, {})
            else:
                manual_features = {
                    "ScreenSize": screen_size,
                    "PPI": ppi,
                    "camera_score": camera_score,
                    "main_camera_mp": main_camera,
                    "value_score": value_score,
                    "NumberOfReview": reviews,
                    # Default values for other required fields
                    "total_resolution": 2430000,
                    "has_telephoto": 1,
                    "has_ultrawide": 1,
                    "popularity_score": 60.0,
                    "price_segment": 1,
                    "has_warranty": 1
                }
                result = app.predict_flexible(services, input_method, "", manual_features)
            
            return format_predictions(result)
        
        def handle_expert_prediction(services, current_user, *expert_features):
            if not current_user:
                return "<div style='color: #e74c3c; padding: 20px; text-align: center;'>Vui lòng đăng nhập để sử dụng tính năng này</div>"
            
            manual_features = {
                "ScreenSize": expert_features[0],
                "PPI": expert_features[1],
                "total_resolution": expert_features[2],
                "camera_score": expert_features[3],
                "main_camera_mp": expert_features[4],
                "num_cameras": expert_features[5],
                "camera_feature_count": expert_features[6],
                "has_telephoto": expert_features[7],
                "has_ultrawide": expert_features[8],
                "has_ois": expert_features[9],
                "popularity_score": expert_features[10],
                "overall_score": expert_features[11],
                "display_score": expert_features[12],
                "camera_rating": expert_features[13],
                "value_score": expert_features[14],
                "price_segment": expert_features[15],
                "is_premium": expert_features[16],
                "has_warranty": expert_features[17],
                "NumberOfReview": expert_features[18]
            }
            
            # Remove None values
            manual_features = {k: v for k, v in manual_features.items() if v is not None}
            result = app.predict_flexible(services, "manual", "", manual_features)
            return format_predictions(result)

        # Bind events
        login_btn.click(handle_login, 
                       inputs=[login_username, login_password],
                       outputs=[current_user_state, login_status, main_tab, user_info_display])
        
        register_btn.click(handle_register,
                          inputs=[reg_username, reg_password, reg_email],
                          outputs=[register_status])
        
        logout_btn.click(handle_logout,
                        outputs=[current_user_state, main_tab, user_info_display, login_status])
        
        input_method.change(toggle_input_method,
                          inputs=input_method,
                          outputs=[product_id, basic_manual_inputs])
        
        quick_predict_btn.click(handle_quick_prediction,
                               inputs=[quick_service, quick_product_id, current_user_state],
                               outputs=quick_output)
        
        advanced_predict_btn.click(handle_advanced_prediction,
                                  inputs=[services, input_method, product_id, current_user_state,
                                         basic_screen_size, basic_ppi, basic_camera_score, 
                                         basic_main_camera, basic_value_score, basic_reviews],
                                  outputs=advanced_output)
        
        expert_predict_btn.click(handle_expert_prediction,
                                inputs=[expert_services, current_user_state,
                                       expert_screen_size, expert_ppi, expert_total_resolution,
                                       expert_camera_score, expert_main_camera, expert_num_cameras,
                                       expert_camera_features, expert_has_telephoto, expert_has_ultrawide,
                                       expert_has_ois, expert_popularity, expert_overall, expert_display_score,
                                       expert_camera_rating, expert_value_score, expert_price_segment,
                                       expert_is_premium, expert_has_warranty, expert_reviews],
                                outputs=expert_output)
        
        # Hướng dẫn sử dụng
        gr.Markdown("---")
        gr.Markdown("""
        ### Hướng Dẫn Sử Dụng
        
        **1. Dự Đoán Nhanh**
        - Sử dụng dữ liệu từ Feature Store
        - Chọn một dịch vụ và nhập Product ID
        - Phù hợp cho đánh giá nhanh sản phẩm có sẵn
        
        **2. Dự Đoán Nâng Cao**  
        - Kết hợp Feature Store và nhập liệu cơ bản
        - Chọn nhiều dịch vụ cùng lúc
        - Phù hợp cho so sánh và phân tích linh hoạt
        
        **3. Nhập Liệu Chuyên Sâu**
        - Nhập đầy đủ thông số kỹ thuật
        - Kiểm soát chi tiết tất cả features
        - Phù hợp cho đánh giá sản phẩm mới hoặc prototype
        
        **Lưu ý:** Tất cả tính năng yêu cầu đăng nhập
        """)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    print("✅ Gradio interface created successfully!")
    print("🌐 Starting server on http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7867,
        share=False
    )