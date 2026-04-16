FROM python:3.10-slim

# Cài đặt các thư viện hệ điều hành cho PyTorch, OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc chính
WORKDIR /app

# Ưu tiên copy requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn backend
COPY app/ ./app/

# Copy model weights
COPY weights/ ./weights/

# Port cho API và Ray Dashboard
EXPOSE 8000
EXPOSE 8265
