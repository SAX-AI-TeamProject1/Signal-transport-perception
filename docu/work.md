# 날짜 별 작업 현황

## 2026.08.01 (git 브랜치 정리, 트랙 연동 인계 문서, 스케일 버그)
- `main`에 커밋(`feat : new label train`, 어제 세션 전체 작업)해두고 `git pull`했더니 `origin/main`이 7개 커밋 앞서있어 merge 충돌 — 무사히 해결(병합 커밋)됐고, `deploy` 브랜치에도 `main`을 머지해 오늘 작업 전부 반영 확인
- `docu/track-follow-integration-guide.md` 신규 작성: `track_follow.py`를 Signal-Simulation `waypoint_follower.py`에 연결하는 인계 명세서. 기존 `yolo-perception-integration-guide.md`(라이다 트리거 장애물 분류용)와 같은 형식으로 작성했고, 그 문서에 남아있던 stale 정보(사람/지게차 메시 없음 — 이미 해소됨)도 6장에 정리
- `best.pt`가 git에 없어서(`.gitignore`, ADR-006) Signal-Simulation이 이 패키지를 pip 설치해서 import할 때 모델을 못 가져오는 문제 확인. 원인 두 가지: (1) 모델 자체가 git에 없음(의도됨, `tools/fetch_model.py`로 Kaggle에서 받아야 함) (2) `infer.py`의 `WEIGHTS` 경로가 "레포 루트에서 실행"을 전제한 상대경로라 다른 레포에서 import하면 깨짐. `src/weights.py` 신규 작성 — `resolve_weights()`가 로컬 dev 경로 우선, 없으면 `~/.cache/signal-transport-perception/`로 폴백, 거기도 없으면 Kaggle에서 자동 fetch. `/tmp`(레포 밖)에서 직접 실행해 end-to-end 검증 완료
- 학습된 모델을 실제 창고 전체 조망 화면(테스트 GUI)에 돌려보니 완전히 실패 — 화면 전체가 `Vehicle` 박스 하나로 뭉개짐. 처음엔 "다중 오브젝트라서"로 오판했으나, 같은 물체를 클러터 없이 단일로 8배 멀리서(작게) 캡처해 재현한 결과 탐지 0개 확인 — 진짜 원인은 **학습 캘리브레이션이 항상 프레임의 30~50%로 물체 크기를 맞춰서, 실배포에서 흔한 "멀리서 작게 보이는" 스케일 자체가 학습 데이터에 없었던 것**(ADR-012)
- `DISTANCE_RATIOS`를 `[0.7, 1.0, 1.6]` → `[0.5, 0.7, 1.0, 1.6, 3.0, 6.0]`로 확장, 전체 10개 메시 재캡처(108→216장/메시, 총 2160장)
- 재캡처 후 재학습했는데 **결과가 이전 학습과 소수점까지 거의 동일하게 나와서** 이상함을 느끼고 조사 — `prepare_dataset.py`의 `write_split()`이 `data/dataset/{train,val}`을 비우지 않고 파일만 추가하는 버그를 발견(이번 세션에서 두 번 실행하며 3328장까지 누적, Kaggle에 올라간 것도 오염됨). `shutil.rmtree`로 복사 전에 비우도록 수정, 재검증(정확히 1728/432 = 2160장) 후 재학습
- 이번엔 결과가 확실히 다르게 나옴(mAP50 0.985→0.995, mAP50-95 0.955→0.981), 8배 축소 재현 테스트도 confidence 0.89로 정상 탐지 — 스케일 문제 해결 확인
- 부수적으로 `kaggle kernels output` CLI가 다운로드는 다 끝냈는데 프로세스가 안 죽고 2시간 넘게 걸려있던 사례 발견 — 임시 폴더에 파일 다 받아져 있으면 죽이고 수동으로 옮기면 됨(처음부터 재시도 불필요)

## 2026.07.31 (트랙 추종)
- navi_factory 월드 스크린샷을 보고 새 요구사항 확인: 로봇이 바닥의 주황색 대시라인 트랙을 보고 자율주행(코너 포함)해야 함. 스크린샷에 원형 패드(출발/적재)·콘+원형 마커(경유지)·직선/대각선 대시라인이 섞여 있어 단순 라인 추종이 아니라 분기점·반환점·목표지점이 있는 실제 라우팅 네트워크임을 확인
- 설계 결정(ADR-011): 인식은 YOLO 대신 HSV 색상 threshold 기반 센터라인 검출(코너에서 부드러운 곡선을 위해), 라우팅(분기 선택/반환/목표 판단)은 Signal-Simulation이 이미 웨이포인트 순서를 알고 있다고 보고 그쪽 책임으로 분리 — 이 레포는 "눈앞 트랙 오프셋+곡률"만 인식
- 여러 차량이 트랙을 가리거나 헷갈리게 하는 문제 제기 → 기존 YOLO 탐지 결과(Vehicle/Person 박스)를 트랙 색상 마스크에서 빼는 방식으로 해결
- `src/track_follow.py` 신규 작성: `track_color_mask`(HSV+장애물 마스킹), `select_own_track`(화면 하단 중앙에 맞닿은 연결성분만 선택 — 무관한 다른 차선 무시), `fit_curve`(가까운~먼 밴드로 나눠 2차 다항식 피팅), `estimate_track`(offset/curvature/confidence + 프레임 간 평활화·분기 오인 억제)
- HSV 임계값은 실제 Gazebo(ogre2) 렌더로 track_ne_entry_1을 스폰해 캡처한 프레임에서 픽셀을 직접 샘플링해 캘리브레이션 — 그림자 진 면도 색상/채도는 유지되고 명도만 낮아짐을 확인
- 합성 이미지로 스트레스 테스트: 트랙에 바로 맞닿아 겹치는 가짜 오브젝트가 있을 때 마스킹 없이는 offset 0.20(틀림)+confidence 1.0(거짓 확신)로 나오다가, obstacle_boxes로 마스킹하면 offset≈0(정답)+confidence 0.46으로 정상화됨을 확인
- 미완료: ROS 2 출력 인터페이스(프로젝트 전체 미정, ADR-005)와 실제 시뮬레이션 라이브 연동 검증은 다음으로 미룸 — 이번엔 순수 인식 함수(이미지→오프셋/곡률)까지만

## 2026.07.31
- Signal-Simulation의 navi_factory 월드 현황 재확인: ADR-007에 "person 모델 없음"으로 미해결 기록했던 것과 달리, 실제로는 `StandingPerson/WalkingPerson/CasualFemale`(사람)과 `Forklift`(차량)가 이미 배치돼 있음을 확인 — 해당 미해결 항목 해소
- 새 클래스를 추가하려면 기존 `MESH_GEOMETRY`의 half-extent 값을 Blender로 측정해야 하는데, 이 프로젝트는 Blender를 안 쓰기로 한 상태(`render_gazebo_assets.py` 자체가 예전에 삭제됨)라 측정 수단이 없다는 문제 제기가 있었음
- gz-sim(Harmonic 8.14)에 내장된 `boundingbox_camera` 센서 + `gz-sim-label-system` 플러그인으로 `src/capture_gazebo_dataset.py`의 라벨링 방식을 통째로 교체(ADR-010): 렌더러가 직접 계산한 픽셀 단위 박스를 그대로 쓰므로 메시 측정이 아예 필요 없어짐. `yolo_label()` 삼각함수 투영, half_x/half_y/half_z, 1.2배 패딩 보정 전부 제거
- 자체 model.sdf가 없는 raw 메시(Person 3종, Forklift)는 navi_factory.sdf가 이미 쓰고 있는 `<visual>` 구성(mesh 경로, pose)을 그대로 재사용해 스폰하도록 `RAW_MESH_MESHES` 신설. `Vehicle` 후보였던 `OGV`는 collision 정의가 아예 없고 인스턴스별 스케일도 제각각이라 이번엔 제외, `Forklift`만 우선 등록
- 클래스를 4개(PalletJack/Bucket/Cluttering/TrashCan) → 6개(+ Person, Vehicle)로 확장. 기존 class_id(0~3)는 그대로 유지
- 검증: `TrashCanC_01`(기존 패키지형 클래스)로 회귀 캡처해 기존과 동등(더 타이트)하게 동작함을 확인, `StandingPerson`/`Forklift`(신규 raw-mesh 클래스)도 각각 캡처해 픽셀 단위로 딱 맞는 박스가 나옴을 시각적으로 확인
- 1차로는 라벨 계산만 자동화하고 카메라 거리/높이(반지름, 중심 높이)는 메시별 상수로 남겨뒀는데, "그게 무슨 자동화냐"는 피드백을 받고 이것도 마저 없앰 — `calibrate_target()`이 boundingbox_camera 피드백만 보고 거리를 좁혀 들어가며 프레임 진입을 확인 → 높이를 세로 중앙에 맞게 보정 → 목표 프레임 비율로 거리 재조정, 이 3단계 루프로 실시간 캘리브레이션. 첫 구현은 높이 보정 부호가 반대라 물체가 하늘 위로 발산하는 버그가 있었음(디버깅 중 발견, 부호 수정 후 재검증 완료)
- 이제 새 클래스 등록에 필요한 정보는 "어떤 메시를 스폰할지"뿐 — `TrashCanC_01`/`Bucket_01`(회귀) + `StandingPerson`/`Forklift`(신규) 전부 재검증 통과
- 전체 10개 메시(6클래스) 정식 재캡처 완료(108장씩 총 1080장). 도중 `camera_low`에서 프레임 수신이 한 번 씹혔는데 재시도 로직이 정상 복구함(하드닝 효과 확인)
- `prepare_dataset.py` 실행 시 1728장으로 잡혀서 확인해보니, 세션 전에 남아있던 접두사 없는 옛 폴더(`Bucket_01`, `ClutteringA_01` 등 6개, 이전 라벨링 방식의 잔재)가 새 접두사 폴더와 같이 잡힌 것 — 옛 폴더 삭제 후 재분할해 정상적으로 1080장(train 862/val 218)으로 정리
- Kaggle 재학습(50epoch) 완료: 전체 mAP50 0.985 / mAP50-95 0.955. 클래스별로는 Bucket/Cluttering/TrashCan/Person/Vehicle 전부 mAP50-95 0.96대 이상(Person이 0.992로 최고 — boundingbox_camera 자동 라벨링 정확도가 그대로 반영된 것으로 보임)인데, **PalletJack만 0.838로 확실히 처짐**
- 학습 스윕에 없던 각도로 스팟체크: Forklift(Vehicle) 0.98/0.98, StandingPerson(Person) 0.95/0.96로 우수. 반면 PalletJack은 confidence 0.75, 다른 각도에선 아예 박스가 2개(0.77+0.39)로 쪼개지는 중복 탐지 발생 — 얇고 튀어나온 파츠(핸들, 포크 두 갈래)가 많은 형태 특성으로 추정. 심각하진 않지만(recall 자체는 학습 시 1.0) 실사용 시 참고 필요 — 개선(각도 다양화 재캡처/재학습)은 이번엔 보류하고 다음으로 넘기기로 함

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

