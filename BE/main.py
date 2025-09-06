from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
import os

# Load trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Class labels 
class_labels = ["plastic", "organic", "metal"]

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FE folder as static files so CSS/JS work
fe_path = os.path.join(os.path.dirname(__file__), "FE")
app.mount("/static", StaticFiles(directory=fe_path), name="static")

# Serve the index.html on /
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(fe_path, "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((64,64))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,64,64,3)

    pred = model.predict(img_array)
    class_index = int(np.argmax(pred[0]))
    confidence = float(np.max(pred[0]))

    return {
        "predicted_class": class_labels[class_index],
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
