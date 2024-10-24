from fastapi import FastAPI, File, UploadFile
from app.models.yolo import YoloModel
from app.utils.draw import draw_boxes_on_image
import cv2
import numpy as np

app = FastAPI()

# Initialize YOLO model
yolo_model = YoloModel(model_path="yolov8n.pt")  # Replace with the correct model

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect seats (occupied and unoccupied) using YOLO
    detections = yolo_model.detect_seats(img)
    
    # Draw bounding boxes on image
    processed_img = draw_boxes_on_image(img, detections)
    
    # Save or return the processed image
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return {"message": "File processed successfully!"}
