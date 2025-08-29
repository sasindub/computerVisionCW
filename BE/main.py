from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

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
