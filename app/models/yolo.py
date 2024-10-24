from ultralytics import YOLO

class YoloModel:
    def __init__(self, model_path: str):
        # Load YOLO model
        self.model = YOLO(model_path)
    
    def detect_seats(self, image):
        # Perform YOLO detection
        results = self.model(image)
        
        detections = []
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())
                class_id = int(bbox.cls[0])
                confidence = bbox.conf[0]
                
                # Filter based on class (customize for seats)
                if class_id in [OCCUPIED_SEAT_CLASS, UNOCCUPIED_SEAT_CLASS]:
                    detections.append({
                        "class_id": class_id,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence
                    })
        return detections
