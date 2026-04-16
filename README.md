OCR Image Project

https://github.com/user-attachments/assets/e1415720-2ea6-45af-8c38-a393db90e4d2

In this project, we will train a YOLOv11 model to detect text regions and use a CRNN model to recognize text.

## 📁 Project Structure

```
ocr/
├── backend/                # Backend API (FastAPI + Ray Serve)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── Makefile
│   ├── app/
│   │   ├── ocr.py          # OCR pipeline (Ray Serve)
│   │   ├── crnn.py          # CRNN model definition
│   │   └── object_detection.py
│   └── weights/             # Model weights
│       ├── best.pt
│       └── ocr_crnn.pt
│
├── frontend/               # Frontend UI (Streamlit)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── app.py           # Streamlit Web UI
│
├── notebooks/              # Jupyter Notebooks (training & experiments)
├── datasets/               # Training data
├── runs/                   # Training runs
├── docker-compose.yml      # Docker orchestration
└── .env.example            # Environment variables template
```

## 🐳 Quick Start (Docker)

```bash
# Build và khởi chạy tất cả services
docker compose up --build

# Chạy ở background
docker compose up --build -d

# Dừng services
docker compose down
```

## 🛠️ Setup Local (Không Docker)

### 1. Tạo môi trường ảo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Khởi động Backend

```bash
cd backend
make init
make deploy_ocr
```

### 3. Khởi động Frontend

```bash
cd frontend/app
streamlit run app.py
```

## 🌐 Access

| Service        | URL                        |
|----------------|----------------------------|
| API Docs       | http://localhost:8000/docs  |
| Ray Dashboard  | http://localhost:8265       |
| Streamlit UI   | http://localhost:8501       |
