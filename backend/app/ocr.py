import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from ray import serve
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from app.crnn import CRNN

app = FastAPI()

# Constants
TEXT_DET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "best.pt"
)
OCR_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "ocr_crnn.pt"
)

# Character set configuration
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# Model configuration
HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, ocr_handle):
        self.handle = ocr_handle

    async def process_image(self, image_data: bytes) -> Response:
        """Common image processing logic for both URL and file upload"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name

            # Request OCR results using the temp file path
            predictions = await self.handle.process_image.remote(temp_file_path)

            # Load the image and draw predictions
            image = Image.open(temp_file_path)
            annotated_image = await self.handle.draw_predictions.remote(
                image, predictions
            )

            # Convert annotated image to bytes
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="PNG")
            file_stream.seek(0)

            # Clean up the temporary file
            os.unlink(temp_file_path)

            return Response(
                content=file_stream.getvalue(),
                media_type="image/png",
                headers={"X-Predictions": str(predictions)},
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    @app.get("/ocr")
    async def ocr_url(self, image_url: str):
        """Endpoint for processing images from URLs"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return await self.process_image(response.content)
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")

    @app.post("/ocr/upload")
    async def ocr_upload(self, file: UploadFile = File(...)):
        """Endpoint for processing uploaded image files"""
        try:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            content = await file.read()
            return await self.process_image(content)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded file: {e}"
            )


@serve.deployment(
    ray_actor_options={"num_gpus": 0, "num_cpus": 1},
    num_replicas=1,
)
class OCRService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_model = YOLO(TEXT_DET_MODEL_PATH).to(self.device)
        self.reg_model = CRNN(
            vocab_size=len(CHARS),
            hidden_size=HIDDEN_SIZE,
            n_layers=N_LAYERS,
            dropout=DROPOUT_PROB,
            unfreeze_layers=UNFREEZE_LAYERS,
        ).to(self.device)
        self.reg_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=self.device))
        self.reg_model.eval()

        # Define transform for inference
        self.transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def text_detection(self, img_path):
        """Detect text regions in the image"""
        results = self.det_model(img_path, verbose=False)[0]
        return (
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.names,
            results.boxes.conf.tolist(),
        )

    def text_recognition(self, img):
        """Recognize text in the cropped image"""
        transformed_image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.reg_model(transformed_image).cpu()
        text = self.decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
        return text

    def process_image(self, img_path: str):
        """Process the image through the OCR pipeline"""
        try:
            # Detect text regions
            bboxes, classes, names, confs = self.text_detection(img_path)

            # Load image
            img = Image.open(img_path)
            predictions = []

            # Process each detection
            for bbox, cls_idx, conf in zip(bboxes, classes, confs):
                x1, y1, x2, y2 = bbox
                name = names[int(cls_idx)]

                # Crop and recognize text
                cropped_image = img.crop((x1, y1, x2, y2))
                transcribed_text = self.text_recognition(cropped_image)
                predictions.append((bbox, name, conf, transcribed_text[0]))

            return predictions

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    def draw_predictions(self, image, predictions):
        """Draw bounding boxes and recognized text above each box."""
        image_array = np.array(image)
        annotator = Annotator(image_array, font="Arial.ttf", pil=False)

        annotated_image = Image.fromarray(annotator.result())
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()

        predictions = sorted(predictions, key=lambda x: x[0][1])

        for bbox, class_name, confidence, text in predictions:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = colors(hash(class_name) % 20, True)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = text or class_name
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = max(0, min(x1, annotated_image.width - text_width - 6))
            text_y = y1 - text_height - 6
            if text_y < 0:
                text_y = min(y1 + 2, annotated_image.height - text_height - 4)

            background_box = [
                text_x,
                text_y,
                text_x + text_width + 6,
                text_y + text_height + 4,
            ]
            draw.rectangle(background_box, fill=color)
            draw.text(
                (text_x + 3, text_y + 2),
                label,
                fill=(255, 255, 255),
                font=font,
            )

        return annotated_image

    def decode(self, encoded_sequences, idx_to_char, blank_char="-"):
        decoded_sequences = []

        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None  # To track the previous character

            for token in seq:
                if token != 0:  # Ignore padding (token = 0)
                    char = idx_to_char[token.item()]
                    # Append the character if it's not a blank or the same as the previous character
                    if char != blank_char:
                        if char != prev_char or prev_char == blank_char:
                            decoded_label.append(char)
                    prev_char = char  # Update previous character

            decoded_sequences.append("".join(decoded_label))

        return decoded_sequences


# ----------------  Create the service
entrypoint = APIIngress.bind(OCRService.bind())
