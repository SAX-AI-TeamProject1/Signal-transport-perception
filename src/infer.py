# Last updated: 2026-07-31

from ultralytics import YOLO

from weights import resolve_weights

def load_model(weights : str | None = None) -> YOLO:
    return YOLO(weights or str(resolve_weights()))

def predict(model : YOLO, source: str):
    return model.predict(source=source)