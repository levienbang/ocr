FROM python:3.10-slim

# Thiết lập thư mục làm việc chính
WORKDIR /app

# Ưu tiên copy requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn frontend
COPY app/ ./app/

# Port cho Streamlit UI
EXPOSE 8501
