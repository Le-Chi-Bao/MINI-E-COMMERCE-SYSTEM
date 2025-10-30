# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary __init__.py files
RUN find /app -type d -name "api" -o -name "web" | xargs -I {} touch {}/__init__.py

# Expose ports
EXPOSE 8000 7869

# Default command (sẽ được override trong docker-compose)
CMD ["echo", "Use docker-compose to start individual services"]