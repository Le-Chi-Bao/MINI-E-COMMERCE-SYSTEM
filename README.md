# 1. Clone và vào thư mục project
cd phone_price_predictor

# 2. Copy file environment
cp .env.example .env
copy .env.example .env

# 3. Chạy với Docker Compose
docker-compose up -d

# 4. Kiểm tra các service
docker-compose ps

docker-compose logs web

# 5. Truy cập ứng dụng
# FastAPI: http://localhost:8000/docs
# Gradio: http://localhost:7860