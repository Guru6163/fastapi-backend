from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...), min_threshold_length: int = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"message": "Error: Unable to open image file"})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_grains = 0
    broken_grains = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        grain_length = max(w, h)
        if grain_length >= min_threshold_length:
            long_grains += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            broken_grains += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    total_grains = long_grains + broken_grains
    long_percentage = (long_grains / total_grains) * 100 if total_grains > 0 else 0
    broken_percentage = (broken_grains / total_grains) * 100 if total_grains > 0 else 0

    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = base64.b64encode(img_encoded).decode('utf-8')

    return {
        "long_percentage": long_percentage,
        "broken_percentage": broken_percentage,
        "image": img_bytes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
