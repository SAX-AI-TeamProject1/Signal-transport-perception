# 날짜 별 작업 현황

## 2026.07.21
- YOLO 프로젝트 폴더 구조(data/, src/) 설계 및 CLAUDE.md 작성
- 데이터셋 소스를 COCO → Roboflow(공장/창고 특화)로 전환 결정, 설계 결정들 docu/decisions.md에 ADR로 기록
- src/prepare_dataset.py 작성: Roboflow warehouse-management(v9, person/forklift/pallet 등 16클래스) 다운로드 파이프라인 완성 (overwrite 미적용·경로 타입 버그 디버깅 포함)
- 행동 판단 로직 설계: person=STOP, vehicle=AVOID, 그 외=크기 기반 판단(라이다 거리 연동), 무게는 클래스 기반 휴리스틱으로 처리
- .gitignore에 data/, runs/, *.pt 추가해 대용량 데이터·산출물 git 제외 처리

