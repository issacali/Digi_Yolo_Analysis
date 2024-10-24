import cv2

# Customize these for your seat detection class IDs
OCCUPIED_SEAT_CLASS = 0  # Example class ID for occupied seats
UNOCCUPIED_SEAT_CLASS = 1  # Example class ID for unoccupied seats

def draw_boxes_on_image(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Define label based on class ID
        label = "Occupied Seat" if class_id == OCCUPIED_SEAT_CLASS else "Unoccupied Seat"
        
        # Draw bounding box
        color = (0, 255, 0) if class_id == OCCUPIED_SEAT_CLASS else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image
