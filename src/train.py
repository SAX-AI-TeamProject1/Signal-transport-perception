# Last updated: 2026-07-28

from pathlib import Path

from ultralytics import YOLO
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = "data/dataset/data.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main() -> None :
    model = YOLO(MODEL)
    model.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        imgsz = IMG_SIZE,
        batch = BATCH,
        device= get_device(),        # GPU 가속
        # ultralytics folds a relative `project` back under its own global runs_dir setting,
        # so this must be absolute to actually land inside the repo.
        project = str(REPO_ROOT / "data" / "runs"),
        name = "train",
    )


if __name__ == "__main__":
    main()

'''

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/50         5.5G      2.442      5.004       1.86        495        640: 1% ──────────── 7/476 4.3s/it 32.2s<33:37

box_loss: 예측한 바운딩박스 위치·크기가 실제 정답 박스와 얼마나 차이 나는지 측정하는 손실. "물체가 어디 있는지, 얼마나 큰지"를 얼마나 정확히 맞추는가에 대한 지표입니다.
cls_loss: 클래스 분류가 얼마나 맞았는지(person인데 forklift라고 하진 않았는지 등)에 대한 손실.
dfl_loss: 바운딩박스 경계를 더 정밀하게 다듬는 데 쓰이는 손실(YOLOv8 특유의 방식).
       
'''