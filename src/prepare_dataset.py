# Last updated: 2026-07-23
'''
capture_gazebo_dataset.py의 결과물을 YOLO train/val 데이터셋으로 분할한다.

data/raw/gazebo_capture/<클래스>/{images,labels}/ (capture_gazebo_dataset.py 참고)를 읽어서
다음을 생성한다:
    data/dataset/{train,val}/{images,labels}/
    data/dataset/data.yaml

리포 루트에서 프로젝트 .venv로 실행 (Gazebo 전용 .venv-gz 아님):
    .venv/bin/python src/prepare_dataset.py
'''

import random
import shutil
from pathlib import Path

import yaml

RAW_DIR = Path("data/raw/gazebo_capture")
DATASET_DIR = Path("data/dataset")
TRAIN_RATIO = 0.8
SEED = 42


# capture_gazebo_dataset.py가 써둔 classes.txt에서 클래스 이름 목록(=클래스 ID 순서)을 읽는다.
def read_class_names() -> list[str]:
    classes_path = RAW_DIR / "classes.txt"
    return classes_path.read_text().strip().splitlines()


# RAW_DIR 밑의 모든 메시별 폴더(_worlds 제외)를 뒤져서 이미지/라벨 쌍을 전부 모은다.
# 여러 메시가 같은 의미 클래스로 합쳐질 수 있어서(capture_gazebo_dataset.py 참고), 폴더명이
# 아니라 라벨 파일에 이미 적힌 class_id로 클래스를 구분한다.
def collect_all_pairs() -> list[tuple[Path, Path, str]]:
    pairs = []
    for mesh_dir in sorted(RAW_DIR.iterdir()):
        if not mesh_dir.is_dir() or mesh_dir.name == "_worlds":
            continue
        images_dir, labels_dir = mesh_dir / "images", mesh_dir / "labels"
        for image_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            class_id = label_path.read_text().split()[0]
            pairs.append((image_path, label_path, class_id))
    return pairs


# class_id별로 섞은 뒤 TRAIN_RATIO 비율로 train/val을 나눈다(클래스마다 비율이 고르게 유지됨).
def split_dataset() -> tuple[list, list]:
    rng = random.Random(SEED)
    by_class: dict[str, list] = {}
    for image_path, label_path, class_id in collect_all_pairs():
        by_class.setdefault(class_id, []).append((image_path, label_path))

    train_pairs, val_pairs = [], []
    for pairs in by_class.values():
        rng.shuffle(pairs)
        cutoff = round(len(pairs) * TRAIN_RATIO)
        train_pairs += pairs[:cutoff]
        val_pairs += pairs[cutoff:]
    return train_pairs, val_pairs


# 이미지/라벨 쌍을 data/dataset/<split>/{images,labels}/로 실제 복사한다.
def write_split(pairs: list[tuple[Path, Path]], split_name: str) -> None:
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


# train.py가 바로 읽을 수 있는 data.yaml(경로/클래스 목록)을 생성한다.
def write_data_yaml(class_names: list[str]) -> None:
    data_yaml = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (DATASET_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))


# 클래스 목록 로드 -> train/val 분할 -> 복사 -> data.yaml 생성까지 전체 파이프라인을 순서대로 실행한다.
def main() -> None:
    class_names = read_class_names()
    train_pairs, val_pairs = split_dataset()
    write_split(train_pairs, "train")
    write_split(val_pairs, "val")
    write_data_yaml(class_names)
    print(f"train: {len(train_pairs)} images, val: {len(val_pairs)} images")
    print(f"data.yaml: {(DATASET_DIR / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
