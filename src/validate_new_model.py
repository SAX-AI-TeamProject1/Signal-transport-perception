# Last updated: 2026-08-04
'''
새로 학습한 가중치를 test_picture/의 원본 스크린샷에 돌려 오탐지(배경 구조물 → Person/Vehicle
등)가 줄었는지 확인하는 1회성 검증 스크립트. 결과는 test_picture/_annotated_new/에 저장한다.

.venv/bin/python src/validate_new_model.py <weights_path>
'''

import sys
from pathlib import Path

from infer import load_model, predict

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PICTURE_DIR = REPO_ROOT / "test_picture"
OUT_DIR = TEST_PICTURE_DIR / "_annotated_new"


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    model = load_model(weights)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(TEST_PICTURE_DIR.glob("*.png")):
        results = predict(model, str(image_path))
        boxes = results[0].boxes
        names = results[0].names
        print(f"\n{image_path.name}:")
        if len(boxes) == 0:
            print("  탐지 없음")
        for box in boxes:
            print(f"  {names[int(box.cls)]} {float(box.conf):.2f}")

        annotated_bgr = results[0].plot()
        from PIL import Image
        Image.fromarray(annotated_bgr[:, :, ::-1]).save(OUT_DIR / image_path.name)

    print(f"\n결과 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
