FROM python:3.9-slim

WORKDIR /app

# ✅ QUAN TRỌNG: Thêm Python path để tìm thấy scripts/
ENV PYTHONPATH=/app

# ✅ TỐI ƯU: Chỉ cài build dependencies, xóa sau khi dùng
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ✅ COPY requirements TRƯỚC để tận dụng cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ TỐI ƯU: Xóa build dependencies sau khi cài packages
RUN apt-get purge -y gcc g++ && apt-get autoremove -y

# ✅ COPY TOÀN BỘ PROJECT
COPY . .

# ✅ TẠO THƯ MỤC CẦN THIẾT
RUN mkdir -p models data/processed data/raw

# ✅ ĐẢM BẢO CÓ __init__.py TRONG SCRIPTS
RUN find /app/scripts -type d -exec touch {}/__init__.py \;

# ✅ KIỂM TRA CẤU TRÚC
RUN echo "✅ Project structure:" && \
    ls -la /app/ && \
    echo "✅ Scripts content:" && \
    ls -la /app/scripts/ && \
    echo "✅ App content:" && \
    ls -la /app/app/

EXPOSE 8000 7860

# ✅ DÙNG uvicorn với reload cho development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# FROM python:3.9-slim

# WORKDIR /app
# ENV PYTHONPATH=/app

# # ✅ TẠM THỜI BỎ GCC - THỬ KHÔNG CẦN COMPILE
# # RUN apt-get update && apt-get install -y --fix-missing gcc g++

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
# RUN mkdir -p models data/processed data/raw
# RUN touch /app/scripts/__init__.py
# RUN ls -la /app/scripts/ && echo "✅ Scripts directory verified"

# EXPOSE 8000 7860
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]