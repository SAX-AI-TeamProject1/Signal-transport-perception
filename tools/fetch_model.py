# Last updated: 2026-07-28

'''
fetch_model.py

재학습 없이, 이미 완료된 kaggle 커널의 마지막 실행 결과물만 받아와 data/runs/train-kaggle/
에 배치한다.

tools/train_kaggle.sh는 "학습 트리거와 완료대기 받기"를 다 하지만 배포 시점엔 이미 학습이
끝나 있으므로 "받기"만 필요하다.
'''

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


KERNEL = "heojaeseong/yolo-percep" # 어느 Kaggle 커널에서 받아올지 
DEST_DIR = Path("data/runs/train-kaggle") # infer.py가 실제로 읽는 경로

def main()->None:
    kaggle_bin = str(Path(sys.executable).parent / "Kaggle") # .venv 안에 설치된 kaggle CLI
    tmp_dir = Path(tempfile.mkdtemp())

    # kenels output : 재학습 안 하고 마지막으로 완료된 실행의 산출물
    subprocess.run([kaggle_bin,"kernels","output",KERNEL,"-p",str(tmp_dir)],check=True)

    DEST_DIR.mkdir(parects=True,exist_ok=True)
    src_run_dir = tmp_dir / "runs" / "train"
    # 받은 파일 중 필요한 3개
    for name in ("weight","result.csv","results.png"):
        target = DEST_DIR / name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(src_run_dir/name),str(target))

    shutil.rmtree(tmp_dir,ignore_errors=True)
    print(f"완료 : {DEST_DIR / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()




















































































