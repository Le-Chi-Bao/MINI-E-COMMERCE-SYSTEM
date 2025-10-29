FROM python:3.9-slim

WORKDIR /app

# ✅ QUAN TRỌNG: Thêm Python path để tìm thấy scripts/
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ COPY TOÀN BỘ PROJECT
COPY . .

RUN mkdir -p models data/processed data/raw

# ✅ ĐẢM BẢO CÓ __init__.py TRONG SCRIPTS
RUN touch /app/scripts/__init__.py

# ✅ KIỂM TRA SCRIPTS TỒN TẠI
RUN ls -la /app/scripts/ && echo "✅ Scripts directory verified"

EXPOSE 8000 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

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