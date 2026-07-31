# Last updated: 2026-07-28

import sys

from ultralytics import YOLO

WEIGHTS = "data/runs/train-kaggle/weights/best.pt"

def load_model(weights : str = WEIGHTS) -> YOLO:
    return YOLO(weights)

def predict(model : YOLO, source: str):
    return model.predict(source=source)