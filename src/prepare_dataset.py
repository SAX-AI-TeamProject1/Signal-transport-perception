# Last updated: 2026-07-23

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# https://universe.roboflow.com/arvist-igsvq/warehouse-management

# ROBOFLOW_API_KEY 값을 환경변수로 등록
load_dotenv("env/roboflow-api.txt")

WORKSPACE = "arvist-igsvq"
PROJECT = "warehouse-management"
VERSION = 9
DOWNLOAD_DIR = Path("data/dataset")

def download_dataset() -> Path:
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download("yolov8",location=str(DOWNLOAD_DIR),overwrite=True)
    return Path(dataset.location)


def main() -> None:
    dataset_path = download_dataset()
    print(f"데이터 셋 다운로드 완료 : {dataset_path}")
    print(f"data.yaml 위치 : {dataset_path/'data.yaml'}")


if __name__ == "__main__":
    main()