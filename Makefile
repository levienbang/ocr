.PHONY: up down build

up:
	docker compose up -d
	@echo ""
	@echo "========================================================"
	@echo "🚀 Hệ thống OCR đã khởi động thành công!"
	@echo "🌐 Giao diện UI (Streamlit): http://localhost:8501"
	@echo "⚙️  Tài liệu API (FastAPI):  http://localhost:8000/docs"
	@echo "📊 Ray Dashboard:            http://localhost:8265"
	@echo "========================================================"

build:
	docker compose up --build -d
	@echo ""
	@echo "========================================================"
	@echo "🚀 Hệ thống OCR đã BUILD và khởi động thành công!"
	@echo "🌐 Giao diện UI (Streamlit): http://localhost:8501"
	@echo "⚙️  Tài liệu API (FastAPI):  http://localhost:8000/docs"
	@echo "📊 Ray Dashboard:            http://localhost:8265"
	@echo "========================================================"

down:
	docker compose down
