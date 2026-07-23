# 날짜 별 작업 현황

## 2026.07.24
- `data/runs/train-kaggle/weights/best.pt`(4클래스, Kaggle 학습본)를 학습 스윕에 없던 각도/거리로 스팟체크: PalletJack(yaw 52°, 3.1배) 0.94 / Cluttering(ClutteringD_01, yaw 110°, 1.9배) 0.96 / Bucket(yaw 200°, 3.4배) 0.95 — 전부 단일·정확한 박스로 검출. 특히 Cluttering은 메시 3종(A/C/D) 중 하나로 테스트해도 잘 잡혀서, 클래스 합치기(ADR-008)가 일반화를 해치지 않았음을 확인
- 검증 이미지를 `docu/spotchecks/2026-07-24_{palletjack,cluttering,bucket}.png`로 리포에 보존(스크래치패드는 세션 종료 시 사라지므로)

## 2026.07.23
- 기존 warehouse-management 모델(person 외 bottle/box 없음) 추론 품질 진단: 실사진(도메인 일치) vs 무관한 사진 비교로 "클래스 부족"이 아니라 "학습 도메인(실사진) ≠ 배포 도메인(Gazebo 시뮬레이션 렌더)" 갭이 근본 원인임을 확인
- COCO+warehouse 혼합, HuggingFace 오픈보캡 오토라벨링 등 대안 검토 후 보류 — 진단 없이 데이터만 늘리는 방향은 기각
- Signal-Simulation의 navi_factory 월드(AWS RoboMaker 창고 에셋) 확인: person/actor 모델이 아직 없음(미해결 항목으로 기록), 동적 오브젝트 6종(PalletJackB/Bucket/ClutteringA·C·D/TrashCanC)만 이 레포의 인식 대상으로 스코프 확정
- 1차 시도: Blender로 6클래스 메시를 배치 렌더링(`render_gazebo_assets.py`)해 216장 확보 → 학습(mAP50 0.995, 자체 검증셋 기준)까지는 성공했으나, 실제 Gazebo 화면으로 검증하니 confidence가 0.98대→0.3대로 폭락, 박스도 중복·헐렁 — 렌더러/조명 도메인 갭이 여전히 남아있음을 실측으로 확인
- 2차 전환: Gazebo 자체 카메라 센서로 직접 캡처하는 `src/capture_gazebo_dataset.py` 신규 작성 — 클래스별 일회용 월드를 이 레포의 `data/raw/gazebo_capture/_worlds/`에 즉석 생성(실제 시뮬레이션에 쓰는 `navi_factory.sdf`는 절대 건드리지 않음), 카메라 고정 + 오브젝트 자체 회전/거리 스윕 방식으로 648장 캡처, pose 기반 자동 라벨링(투영 계산) 구현
- Gazebo 파이썬 바인딩용 별도 인터프리터(`.venv-gz`, --system-site-packages) 구성, `assimp`/Blender 관련 산출물(`data/raw/sim_render`, 구 `data/dataset`, `data/runs/train2`, `render_gazebo_assets.py`, `class_aliases.yaml`) 및 Signal-Simulation에 잘못 만들었던 임시 캡처 월드 전부 삭제해 폴더 정리
- `.vscode/tasks.json`에 "Gazebo 데이터셋 캡처" 태스크 신설 및 번호 재정렬, `train.py`의 저장 경로 버그(ultralytics가 상대경로 project를 자체 runs_dir 밑에 잘못 접어넣는 문제) 절대경로로 수정
- `docu/decisions.md`에 ADR-007(학습 데이터 소스: 실사진 → Gazebo 자체 렌더링) 기록, CLAUDE.md 최신화
- 새 648장 데이터셋(6클래스)으로 재학습 완료: mAP50 0.995 / mAP50-95 0.93, 전부 precision·recall 0.99+. 학습 스윕에 없던 임의 각도로 새로 캡처한 프레임 스팟체크에서 단일·정확한 박스에 confidence 0.95 — Blender 학습본이 같은 테스트에서 0.27~0.46 + 중복박스였던 것과 비교해 도메인 갭이 실제로 해소됐음을 확인
- 클래스 taxonomy 정리: `Bucket_01`처럼 에셋 파일명(버전 접미사 포함)을 그대로 클래스로 쓰는 대신, `ClutteringA/C/D_01`(전부 "잡동사니 더미" 개념, ADR-004상 어차피 크기 기반 판단으로 동일 처리)을 하나의 `Cluttering` 클래스로 합쳐 **6클래스 → 4클래스(PalletJack/Bucket/Cluttering/TrashCan)**로 재정리. `capture_gazebo_dataset.py`를 메시(물리적 3D 자산)와 의미 클래스(라벨)를 분리하는 구조로 리팩터링, `prepare_dataset.py`도 폴더명이 아니라 라벨 파일의 class_id로 그룹핑하도록 수정
- 로컬 재학습이 오래 걸린다는 피드백으로 학습을 Kaggle로 전환: `kaggle` CLI 설치, 기존 노트북(`heojaeseong/yolo-percep`, 원래 Roboflow 다운로드 방식)을 새 4클래스 데이터셋(Kaggle Dataset `heojaeseong/yolo-gazebo-capture`로 업로드) 기반으로 교체. 첫 시도는 data.yaml의 `path`가 로컬 맥 절대경로로 박혀있어서 실패 — 노트북에서 Kaggle 입력 경로로 재작성하도록 고쳐서 재시도 성공
- **최종 Kaggle 학습 결과**(T4 GPU, 약 7분 — 로컬 32분 대비 대폭 단축): mAP50 0.995 / mAP50-95 0.937, 클래스별 mAP50-95는 PalletJack 0.883 / Bucket 0.953 / Cluttering 0.936 / TrashCan 0.976. 체크포인트를 `data/runs/train-kaggle/weights/best.pt`로 반입, 구버전 6클래스 로컬 학습본(`data/runs/train-2`)은 삭제
- (보안 메모) Kaggle 토큰을 대화창에 붙여넣은 사건 발생 — 즉시 폐기 안내. 이후 kaggle CLI가 예전 `kaggle.json` 방식이 아니라 `~/.kaggle/access_token` 단일 토큰 방식으로 바뀐 걸 확인, 파일 간 복사만으로 인증 처리(대화창엔 값 노출 안 함)
- `prepare_dataset.py`, `capture_gazebo_dataset.py`의 설명 주석을 한글 + `'''` 스타일로 통일(앞으로 이 컨벤션 유지)

## 2026.07.21
- YOLO 프로젝트 폴더 구조(data/, src/) 설계 및 CLAUDE.md 작성
- 데이터셋 소스를 COCO → Roboflow(공장/창고 특화)로 전환 결정, 설계 결정들 docu/decisions.md에 ADR로 기록
- src/prepare_dataset.py 작성: Roboflow warehouse-management(v9, person/forklift/pallet 등 16클래스) 다운로드 파이프라인 완성 (overwrite 미적용·경로 타입 버그 디버깅 포함)
- 행동 판단 로직 설계: person=STOP, vehicle=AVOID, 그 외=크기 기반 판단(라이다 거리 연동), 무게는 클래스 기반 휴리스틱으로 처리
- .gitignore에 data/, runs/, *.pt 추가해 대용량 데이터·산출물 git 제외 처리

