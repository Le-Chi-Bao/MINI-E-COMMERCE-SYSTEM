# Dừng và xóa containers cũ
docker-compose down

# Build lại với requirements mới
docker-compose build --no-cache

# Chạy lại
docker-compose up -d

# Kiểm tra logs API
docker-compose logs api

# Kiểm tra logs web
docker-compose logs web