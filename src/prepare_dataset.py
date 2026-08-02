# Last updated: 2026-08-02
'''
capture_gazebo_dataset.py의 결과물을 YOLO train/val 데이터셋으로 분할한다.

data/raw/gazebo_capture/<클래스>/{images,labels}/ (capture_gazebo_dataset.py 참고)를 읽어서
다음을 생성한다:
    data/dataset/{train,val}/{images,labels}/
    data/dataset/data.yaml

분할이 끝나면 곧바로 전체 데이터셋을 Kaggle Dataset으로 업로드한다(한 장씩이 아니라 한 번에
통째로 — 캡처 중간에 조금씩 올리면 네트워크 왕복 때문에 느려지고, train/val 분할도 전체를
다 모아야 가능해서 어차피 끝나고 한 번에 올려야 한다). 업로드를 건너뛰려면 --skip-upload.

리포 루트에서 프로젝트 .venv로 실행 (Gazebo 전용 .venv-gz 아님):
    .venv/bin/python src/prepare_dataset.py [--skip-upload]
'''

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

RAW_DIR = Path("data/raw/gazebo_capture")
DATASET_DIR = Path("data/dataset")
TRAIN_RATIO = 0.8
SEED = 42
KAGGLE_DATASET_SLUG = "heojaeseong/yolo-gazebo-capture"


# capture_gazebo_dataset.py가 써둔 classes.txt에서 클래스 이름 목록(=클래스 ID 순서)을 읽는다.
def read_class_names() -> list[str]:
    classes_path = RAW_DIR / "classes.txt"
    return classes_path.read_text().strip().splitlines()


# RAW_DIR 밑의 모든 메시별 폴더(_worlds 제외)를 뒤져서 이미지/라벨 쌍을 전부 모은다.
# 여러 메시가 같은 의미 클래스로 합쳐질 수 있어서(capture_gazebo_dataset.py 참고), 폴더명이
# 아니라 라벨 파일에 이미 적힌 class_id로 클래스를 구분한다.
#
# _multi_scene(멀티오브젝트 씬, capture_gazebo_dataset.py의 capture_multi_scenes 참고)만 예외 —
# 라벨 파일 한 장에 서로 다른 class_id가 여러 줄 섞여 있을 수 있어서 "첫 줄 class_id"로 묶는 게
# 의미가 없다. 이런 파일은 "multi_scene"이라는 별도 그룹 키 하나로 묶어서 그 그룹 안에서만
# train/val 비율을 지킨다(개별 class_id별 균형은 신경 쓰지 않음 — 이미 단일 오브젝트 캡처가
# 클래스별 균형을 잡아주고 있어서, 멀티오브젝트 씬은 "몇 개가 동시에 나오는가" 다양성이 목적).
def collect_all_pairs() -> list[tuple[Path, Path, str]]:
    pairs = []
    for mesh_dir in sorted(RAW_DIR.iterdir()):
        if not mesh_dir.is_dir() or mesh_dir.name == "_worlds":
            continue
        images_dir, labels_dir = mesh_dir / "images", mesh_dir / "labels"
        is_multi_scene = mesh_dir.name == "_multi_scene"
        for image_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            class_id = "multi_scene" if is_multi_scene else label_path.read_text().split()[0]
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


# 이미지/라벨 쌍을 data/dataset/<split>/{images,labels}/로 실제 복사한다. 매번 지우고 새로
# 채운다 — 안 지우면 이전 실행에서 남은 파일(다른 idx 범위, 다른 train/val 배정)이 계속 쌓여서
# Kaggle에 올라가는 zip까지 오염된다(실제로 겪음: 재캡처 후에도 옛 스케일 이미지가 안 없어져서
# 재학습 결과가 이전과 거의 동일하게 나왔었음).
def write_split(pairs: list[tuple[Path, Path]], split_name: str) -> None:
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"
    shutil.rmtree(images_dir, ignore_errors=True)
    shutil.rmtree(labels_dir, ignore_errors=True)
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


# data/dataset/ 전체(train.zip, val.zip, data.yaml)를 Kaggle Dataset 하나로 통째로 업로드한다.
# 이미 있으면 새 버전으로 갱신(version), 처음이면 새로 생성(create) — 둘 다 시도해서 판단한다.
def upload_to_kaggle() -> None:
    metadata_path = DATASET_DIR / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({
        "title": "yolo-gazebo-capture",
        "id": KAGGLE_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    kaggle_bin = str(Path(sys.executable).parent / "kaggle")  # .venv/bin/python 옆의 kaggle CLI

    version_result = subprocess.run(
        [kaggle_bin, "datasets", "version", "-p", str(DATASET_DIR), "-m", "auto-update from prepare_dataset.py",
         "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if version_result.returncode == 0:
        print("Kaggle Dataset 버전 갱신 완료:", KAGGLE_DATASET_SLUG)
        return

    create_result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if create_result.returncode != 0:
        print("Kaggle 업로드 실패:", version_result.stderr.strip(), create_result.stderr.strip(), file=sys.stderr)
        return
    print("Kaggle Dataset 새로 생성 완료:", KAGGLE_DATASET_SLUG)


# 클래스 목록 로드 -> train/val 분할 -> 복사 -> data.yaml 생성 -> Kaggle 업로드까지 순서대로 실행한다.
def main() -> None:
    class_names = read_class_names()
    train_pairs, val_pairs = split_dataset()
    write_split(train_pairs, "train")
    write_split(val_pairs, "val")
    write_data_yaml(class_names)
    print(f"train: {len(train_pairs)} images, val: {len(val_pairs)} images")
    print(f"data.yaml: {(DATASET_DIR / 'data.yaml').resolve()}")

    if "--skip-upload" not in sys.argv:
        upload_to_kaggle()


if __name__ == "__main__":
    main()
