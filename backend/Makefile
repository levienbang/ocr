.PHONY: init deploy clean streamlit

clean:
	ray stop || true
	rm -rf /tmp/ray_project/*

init: clean
	mkdir -p /tmp/ray_project
	ray start --head --disable-usage-stats --temp-dir=/tmp/ray_project --dashboard-host=0.0.0.0

deploy:
	@echo "Starting object detection server..."
	cd app && serve run object_detection:entrypoint --address="auto"

deploy_ocr:
	@echo "Starting OCR server..."
	cd app && serve run ocr:entrypoint --address="auto"
