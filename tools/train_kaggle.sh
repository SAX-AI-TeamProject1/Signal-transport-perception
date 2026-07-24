#!/usr/bin/env bash
# Last updated: 2026-07-24
# kaggle/yolo-percep 노트북을 Kaggle GPU에서 실행하고, 완료될 때까지 상태를 폴링한 뒤
# 결과(weights, results.csv/png)를 data/runs/train-kaggle/로 정리해서 받아온다.
set -e

KAGGLE="${KAGGLE_BIN:-.venv/bin/kaggle}"
KERNEL="heojaeseong/yolo-percep"

"$KAGGLE" kernels push -p kaggle/yolo-percep
echo "Kaggle 학습 시작됨, 완료까지 대기 중..."

while true; do
    status=$("$KAGGLE" kernels status "$KERNEL" 2>&1)
    echo "$status"
    case "$status" in
        *COMPLETE*) break ;;
        *ERROR*) echo "학습 실패"; exit 1 ;;
    esac
    sleep 20
done

tmp=$(mktemp -d)
"$KAGGLE" kernels output "$KERNEL" -p "$tmp"

rm -rf data/runs/train-kaggle
mkdir -p data/runs/train-kaggle
mv "$tmp/runs/train/weights" "$tmp/runs/train/results.csv" "$tmp/runs/train/results.png" data/runs/train-kaggle/
rm -rf "$tmp"

echo "완료: data/runs/train-kaggle/weights/best.pt"
