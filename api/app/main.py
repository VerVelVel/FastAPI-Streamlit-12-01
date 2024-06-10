import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import PIL
from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
from utils.model_func import load_model, clean_text, load_tokenizer, transform_image, load_yolo
import torch
import io
import cv2

yolo = None
model = None 
tokenizer = None
app = FastAPI()

# Класс для хранения информации о bounding box
class Box(BaseModel):
    xyxy: list
    conf: float
    cls: int

# Класс для хранения результата детекции объектов
class DetectionResult(BaseModel):
    boxes: list[Box]
    names: dict[int, str]


@app.get('/')
def return_info():
    return 'Hello FastAPI'

# Загрузка модели и токенизатора при запуске сервера
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, yolo
    model = load_model()
    tokenizer = load_tokenizer()
    yolo = load_yolo()


# Определение модели для входных данных
class TextData(BaseModel):
    text: str

class TextResults(BaseModel):
    class_pred: str
    probability: float

from fastapi import HTTPException

@app.post('/clf_text', response_model=TextResults)
def clf_text(data: TextData):
    try:
        text = clean_text(data.text)
        tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=100,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']    

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        prediction = torch.sigmoid(output).item()
        if prediction > 0.5:
            class_pred = 'TOXIC'
        else:
            class_pred = 'healthy'

        result = TextResults(class_pred=class_pred, probability=prediction)

        return result
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal server error")



##### run from api folder:
##### uvicorn app.main:app


# FastAPI для детекции объектов на изображении
@app.post('/detect')
async def detect_objects(file: UploadFile):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    detections = yolo.predict(image)

    boxes = []
    for box in detections[0].boxes:
        boxes.append(Box(xyxy=box.xyxy.tolist(), conf=box.conf.item(), cls=int(box.cls)))
    return DetectionResult(boxes=boxes, names=yolo.names)

