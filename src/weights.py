# Last updated: 2026-07-31
'''
best.pt 위치를 정하고, 없으면 Kaggle에서 받아온다.

이 레포 루트에서 실행할 때는 기존처럼 data/runs/train-kaggle/weights/best.pt를 쓴다(ADR-006).
하지만 이 레포를 다른 레포(Signal-Simulation 등)에 pip 설치해서 import하면 실행 위치가 이
레포 루트가 아니므로 그 상대경로가 존재하지 않는다 — 이 경우 CWD와 무관한 고정 캐시 경로로
폴백하고, 거기에도 없으면 Kaggle 커널(heojaeseong/yolo-percep)의 마지막 완료 결과물을 자동으로
받아온다. data/는 git에 올라가지 않으므로(재현 가능 산출물, ADR-006) 이게 사실상 유일한 배포
경로다.
'''

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = "heojaeseong/yolo-percep"
LOCAL_WEIGHTS = Path("data/runs/train-kaggle/weights/best.pt")  # 이 레포 루트에서 실행할 때
CACHE_DIR = Path(
    os.environ.get("SIGNAL_PERCEPTION_CACHE", str(Path.home() / ".cache" / "signal-transport-perception"))
)
CACHE_WEIGHTS = CACHE_DIR / "weights" / "best.pt"


def _fetch_to_cache() -> Path:
    kaggle_bin = str(Path(sys.executable).parent / "kaggle")
    tmp_dir = Path(tempfile.mkdtemp())
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run([kaggle_bin, "kernels", "output", KERNEL, "-p", str(tmp_dir)], check=True, env=env)

    src_run_dir = tmp_dir / "runs" / "train"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("weights", "results.csv", "results.png"):
        target = CACHE_DIR / name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(src_run_dir / name), str(target))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return CACHE_WEIGHTS


def resolve_weights(force_refresh: bool = False) -> Path:
    '''best.pt 경로를 찾아 반환한다 — 로컬 dev 경로(repo root 실행) 우선, 없으면 캐시, 캐시에도
    없으면 Kaggle에서 자동으로 받아와 캐시에 채운 뒤 그 경로를 반환한다.'''
    if LOCAL_WEIGHTS.exists() and not force_refresh:
        return LOCAL_WEIGHTS
    if CACHE_WEIGHTS.exists() and not force_refresh:
        return CACHE_WEIGHTS
    return _fetch_to_cache()
