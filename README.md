# Restaurant Seat Detection API

This project detects the number of occupied and unoccupied seats in restaurant images or videos using the YOLO object detection model and FastAPI.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI app:
```bash
uvicorn app.main:app --reload
```

3. Test the API with an image upload to the `/upload/` endpoint.
