# 날짜 별 작업 현황

## 2026.08.04 (배경/하드 네거티브 캡처 + 재학습, Claude 자동 진행)
- 전날 밤 `test_picture/`에 저장해둔 실제 navi_factory 조감 스크린샷 4장에서 배경 구조물(나무
  기둥, `ControlConsole`, `ElectricalPanelBareModel`, 선반 뒷면)이 Person/Vehicle/PalletJack으로
  오탐지되는 문제를 확인 — `capture_gazebo_dataset.py`에 이미 반영돼 있던(미검증 상태) 수정사항으로
  원인/해결 방향 판단: 학습 데이터의 배경 클러터가 `Shelf/Wall` 두 종류뿐이라 실제 창고의 다른
  소품(콘솔류)·건물 구조(기둥)를 한 번도 본 적이 없었던 것(가장 가까운 학습 클래스로 욱여넣어
  분류)
- 코드에 이미 있던 수정사항 실행: `HARD_NEGATIVE_MESHES`(ControlConsole/ElectricalPanelBareModel,
  라벨 없이 노출), `WAREHOUSE_BACKDROP_SDF`(navi_factory 건물 메시를 배경으로 복제), 조감
  `OVERVIEW_CAMERA` 추가, `MaleVisitor`/`FemaleVisitorWalk` 사람 메시 추가, `NUM_SCENES` 150→250
- 멀티오브젝트 씬 250개 재캡처(카메라 4대, 878장) — `.venv-gz`로 정상 완료 확인(오브젝트/배경/하드
  네거티브 겹침 없이 렌더링됨, 라벨 파일도 하드 네거티브는 라벨 없이 배경으로만 나옴을 확인)
- `prepare_dataset.py` 재분할(train 2661 / val 665) + Kaggle Dataset 업로드, `kaggle/yolo-percep`
  재학습(50 epoch): 전체 mAP50 0.946 / mAP50-95 0.887 (직전 08-03 세션의 0.968/0.922보다 더
  낮아짐 — 배경 다양성이 더 늘어난 만큼 자연스러운 트레이드오프로 추정, 이전에도 반복된 패턴)
- `test_picture/` 원본 스크린샷 4장으로 재검증: 기둥/콘솔/선반을 Person·Vehicle·PalletJack으로
  잘못 잡던 오탐지는 확실히 사라짐(이미지당 박스 수 8~11개 → 2~4개, 실제 지게차/박스더미 등은
  정확한 클래스+타이트한 박스로 검출: Vehicle 0.94, Cluttering 0.80 등)
- **새 문제 발견**: 검증 4장 중 3장에서 "Person" 박스 좌표가 이미지 전체 크기에 거의 근접(예:
  1370x824 이미지에서 [0.4, 2.3, 1370.0, 824.0]) — 위치를 못 찾고 "화면 어딘가에 사람 있음"으로
  뭉뚱그리는 저품질 예측. 처음엔 `.plot()` 렌더링만 보고 실제 사람 위치에 맞는 타이트한 박스로
  착각했으나, `box.xyxy` 좌표를 직접 찍어보고 발견함(스팟체크 시 렌더링 이미지만으로 판단하지
  말고 좌표도 같이 확인할 것). 각 장면에 실제 사람이 있어 클래스 자체는 맞고 person=STOP은
  크기 무관(ADR-004)이라 당장 행동 결정에는 영향 없지만, 위치 추정으로는 쓸모없고 confidence만
  으로는(0.31~0.87, 정상 탐지와 겹침) 걸러지지 않음 — 박스 면적비 기반 필터 등 후처리가 필요해
  보임. 원인/해결 방향은 다음 세션에서 결정
- 새 가중치는 `data/runs/train-kaggle/weights/`에 반영, 이전 가중치는 `data/runs/train-kaggle-prev/`
  에 백업(롤백 대비, 확정되면 정리 필요)

### 후속: 풀프레임 라벨 버그 원인 규명 + 필터링 재학습 (같은 날, 이어서)
- 위 "새 문제"의 원인을 더 파고듦 — 전체 라벨을 스캔해보니 멀티오브젝트/단일메시 가리지 않고
  전체 라벨의 ~4%가 w>0.95 및 h>0.95(사실상 화면 전체)였고, 그중 다수가 서로 무관한 장면인데도
  `0.499219 0.499219 0.998437 0.998437`처럼 **완전히 동일한 좌표값**을 반복 — 실제로 작게 찍힌
  사람이 있는 이미지(`scene_0011_camera_overview.png`)를 직접 열어봐도 라벨은 화면 전체를 가리켜,
  자연스러운 "크고 가까운 물체"가 아니라 `boundingbox_camera`가 투영에 실패했을 때 프레임 전체로
  폴백하는 렌더러/플러그인 버그로 결론
  - 원인 자체(gz-sim 내부)를 고치는 대신, `prepare_dataset.py`에 `MAX_BOX_FRACTION=0.95` 필터를
    추가해 이런 라벨이 하나라도 섞인 이미지-라벨 쌍은 통째로 학습에서 제외(라벨 한 줄만 지우면
    "이 물체는 배경"이라고 잘못 가르치게 되므로 이미지 전체를 버리는 쪽을 선택)
  - 재분할(2661/665 → 필터 후 2532/634, 160장 제외) 후 Kaggle 재학습(kernel v13): mAP50 0.956 /
    mAP50-95 0.899로 직전 세션(0.946/0.887)보다 오히려 개선
  - `test_picture/` 4장 재검증(이번엔 `.plot()` 대신 `box.xyxy` 원좌표 + 프레임 대비 비율을 직접
    출력해서 확인) — **Person 풀프레임 아티팩트는 사라짐**, 4장 모두 Person 박스가 실제 사람
    위치에 맞게 타이트해짐(예: 0.70/0.67/0.64/0.31 confidence, 전부 정상 크기)
  - 다만 **같은 패턴이 낮은 confidence의 PalletJack으로 옮겨 나타남**: 2/4 이미지에서 화면
    테두리를 그대로 따라가는 PalletJack 박스 발견(frac=(1.00,0.92) conf 0.34, frac=(0.92,0.79)
    conf 0.55) — 다만 이번엔 라벨 스캔으로 원인이 다름을 확인: PalletJack 학습 라벨 362개 중
    w>0.8&h>0.8인 것은 10개뿐이고, 전부 `aws_robomaker_warehouse_PalletJackB_01` 단일메시 캡처의
    근접 프레임들(0.83~0.99 사이에서 연속적으로 변화, 위 버그 특유의 완전 동일값 반복 패턴
    없음)이라 **라벨 오염이 아니라 "근접 촬영 시 PalletJack이 프레임을 가득 채운다"는 정상
    학습 데이터**로 보임 — 모델이 배경이 복잡해 확신이 안 설 때 이 근접-풀프레임 패턴을
    끌어다 쓰는 것으로 추정(저confidence 0.34~0.60, 두 이미지 모두 실제 Person이 더 높은
    confidence로 검출돼 있어 현재 시나리오에서 최상위 판정을 뒤집지는 않음)
  - 결론: 이번 세션에서 사용자가 지적한 "Person이 애매하게 잡힘" 문제는 해결. PalletJack
    저confidence 풀프레임 오탐은 별도의 기존 약점(ADR-010에서부터 반복 언급된 PalletJack
    confidence 편차)의 연장으로 보이며, 지금 당장 판정을 뒤집진 않지만 다음 세션에서 검토할
    오픈 아이템으로 남김(단일메시 근접 캡처 프레임 비율을 낮추거나, 후처리 confidence
    threshold를 두는 방향 검토)
  - 가중치는 다시 `data/runs/train-kaggle/weights/`에 반영, 라운드1 가중치는
    `data/runs/train-kaggle-prev2/`에 백업

### 후속 2: navi_factory 실제 트랙 따라 캡처 (같은 날, 이어서)
- 위 두 라운드(멀티오브젝트 씬 + 풀프레임 필터링) 모두 여전히 "합성 월드"에서 캡처한 데이터라
  실제 배포 월드(navi_factory)와의 도메인 갭이 근본적으로 안 없어짐. 사용자 확인("트랙은 판단
  안해도되고 전방 물체가 사람인지 차량인지 등만 판단하면 됨, 오판단 금지")에 따라 이번엔 아예
  navi_factory.sdf를 실제로 띄운 채(read-only, 절대 수정 안 함) `config/tracks.yaml`의 실제 주행
  경로를 따라 가상 카메라를 텔레포트시키며 캡처하는 방식으로 전환
- `src/capture_navi_factory_dataset.py` 신규 작성: `/world/{world}/create`로 카메라 리그를 런타임
  스폰, `/world/{world}/set_pose`로 트랙 경로(1.2m 간격 리샘플) 위를 따라 이동시키며 프레임 캡처.
  라벨은 `boundingbox_camera`(우리가 스폰한 오브젝트에만 동작, navi_factory에 이미 박혀있는
  오브젝트는 못 잡음) 대신 카메라 pose+오브젝트 pose+half-extent로 직접 핀홀 투영해서 계산
  (occlusion은 미반영 — 파일럿 시각 검증으로 허용 가능 수준 확인)
- **주요 발견**: `/world/{world}/dynamic_pose/info`(및 `pose/info`)가 gz-sim `<actor>`(스켈레톤
  애니메이션 워킹 액터) 엔티티의 포즈를 아예 리포트하지 않음(SceneBroadcaster가 액터를 ECM 레벨
  에서 추적 안 함) — 라이브 쿼리 방식을 포기하고, navi_factory.sdf에 박혀있는 8개 액터의
  `<trajectory>` 웨이포인트(시간별 x/y/yaw)를 직접 읽어와 `/world/{world}/stats`의 sim_time
  기준으로 gz-sim과 동일한 보간 로직을 외부에서 재구현(`actor_pose_at()`)해서 해결 — 실제 렌더링된
  사람 위치와 계산된 박스가 일치함을 크롭 확대로 검증
- PalletJack 스케일: navi_factory의 인라인 모델이 `_x15` 메시(자체 스케일 1.5)를 쓰면서 SDF에
  `<scale>1.5</scale>`을 또 얹어놔서 처음엔 1.5×1.5=2.25배 중첩을 의심했으나, `_x15` 폴더의
  model.sdf 자체가 "베이스 메시 대비 1.5배"라는 뜻이라 중첩이 아니라 최종 1.5배가 맞음을
  `_x15`/베이스 두 model.sdf 직접 diff로 확인
- 188장 캡처(PalletJack 2대 고정 위치 + hand_signal_person 1명 고정 + 액터 8명, 라벨 없는 순수
  배경 프레임 52장은 하드 네거티브로 의도적으로 보존) — PalletJack 70박스, Person 366박스
- `prepare_dataset.py`의 멀티라벨 그룹핑을 `_multi_scene` 전용에서 `MULTI_LABEL_DIRS =
  {"_multi_scene", "_navi_factory"}`로 일반화, 빈 라벨(배경 hard negative)도 학습에 포함되도록
  단일메시 폴더와 다르게 스킵 안 함. 분할 결과 train 2682/val 672(+188, 기대치와 정확히 일치)
- **Kaggle 데이터셋 처리 지연 함정 발견**: `kaggle datasets version`이 "완료" 메시지를 반환해도
  서버 쪽 zip 처리가 비동기라, 그 직후 바로 `kaggle kernels push`하면 새 버전이 아직 "ready"가
  안 된 상태라서 커널이 **이전 버전**을 그대로 마운트해 학습해버림(v14 커널: 학습 로그에
  `train/labels... 2532`로 찍혀 이전 데이터임을 뒤늦게 확인 — `test_picture/` 재검증 결과가
  prev3 백업 모델과 confidence 소수점까지 완전히 동일해서 발견함). `kaggle datasets status
  <slug>`로 "ready" 확인 후 커널을 다시 push(v15)해서 재학습, 로그에 `train/labels... 2682`,
  `val/labels... 672`로 새 데이터가 실제로 반영됐음을 확인 — **앞으로는 version 갱신 후 무조건
  `kaggle datasets status`로 ready를 확인하고 나서 커널을 push할 것**
  - 교훈: 결과가 이전과 의심스러울 만큼 똑같으면(수치가 소수점까지 일치) "우연"으로 넘기지 말고
    가중치 md5 비교 + 학습 로그의 실제 이미지 스캔 카운트부터 확인할 것(이번에 이 방법으로 바로
    원인 특정함)
- v15 재학습 결과: mAP50 0.94 / mAP50-95 0.87 (배경 다양성 증가로 인한 자연스러운 소폭 하락,
  이전에도 반복된 패턴)
- `test_picture/` 4장 재검증: 실제 근접 인물은 여전히 정확히 검출(Person 0.87), 지게차를
  Vehicle로 정확히 검출(이전엔 배경 오탐뿐이었음), **더 이상 존재하지 않는 PalletJack을 화면에서
  만들어내는 오탐이 사라짐**(이전 라운드까지 반복되던 문제). 다만 이미지 1장의 기계/기둥 구조물에
  낮은 confidence(0.42)의 Person 오탐이 하나 남아있어 완전 해결은 아님 — 다음 세션에서 계속 관찰
- 새 가중치는 `data/runs/train-kaggle/weights/`에 반영, 이전(이번 라운드 이전) 가중치는
  `data/runs/train-kaggle-prev3/`에 백업

### 후속 3: 남은 두 문제(기계 오탐 + 원거리 미검출) 확실히 해결 (같은 날, 이어서)
- 사용자가 자신의 데스크탑 스크린샷(내가 만든 주석 이미지)을 직접 보고 "사람아닌것도 사람으로
  잡거나 멀리서있는 사람은 잘 못잡아"라고 재확인, "이번에는 확실하게 판단되도록 해줘 제발"이라고
  명확히 요청 — 부분적 패치가 아니라 결정적 조치를 원한다는 의사로 판단
- 옵션 A(재캡처+재학습) vs B(추론 imgsz만 올리기) 검토: v15(imgsz=640 학습)에 대해
  `model.predict(imgsz=960)`으로 스윕 테스트해보니 기계 오탐 1건은 없어지지만, 대신 배경 기둥이
  화면의 거의 100%(frac_h≈0.99)를 채우는 **새로운 더 심각한 오탐**이 여러 이미지에서 나타나고,
  다른 이미지에서는 기존 정탐(사람)이 아예 사라지는 회귀도 발생 — 추론 해상도만 올리는 건
  깨끗한 해법이 아니라고 결론, 학습 해상도 자체를 올리고 데이터도 보강하는 결합 조치로 결정
- `kaggle/yolo-percep/yolo-percep.ipynb`: `IMG_SIZE` 640→960, `BATCH` 16→8(고해상도 OOM 방지)
- `src/capture_navi_factory_dataset.py` 대폭 보강:
  - `STEP_DISTANCE` 1.2→1.0(트랙 샘플링 밀도 증가)
  - **side-glance 캡처**: `SIDE_GLANCE_EVERY`(5지점마다) 지점에서 진행 방향 기준 좌/우 90도로
    고개를 돌린 프레임을 추가 촬영 — 벽/기둥/콘솔 같은 배경 구조물이 화면을 가득 채우는 상황을
    라벨 없는(hard negative) 프레임으로 명시적으로 학습시켜 "화면을 가득 채운 세로로 긴 물체 →
    Person" 오탐 패턴에 대응
  - **멀티패스 캡처**: `NUM_PASSES=2`로 트랙을 두 번 돌림 — gz-sim 시뮬레이션 시계(sim_time)가
    액터 걷기 주기와 정확히 안 맞아떨어지는 특성을 이용해, 같은 카메라 지점이라도 패스마다 액터
    위치/거리 조합이 달라져 원거리 사람 포즈 다양성을 늘림
  - 12지점 파일럿으로 먼저 검증(side-glance가 배경만 라벨 없이 찍히는지, 원거리 액터가
    제대로 잡히는지) 후 전체 실행
  - 전체 캡처 결과: 224지점 × 2패스 = 628장(기존 188장 대비 3.3배) — 기존 `_navi_factory/` 삭제 후
    완전히 새 데이터로 교체(병합 아님)
- `prepare_dataset.py` 재실행: train 3034 / val 760 (기존 2682/672에서 새 데이터만큼 증가,
  산술 검증: 3034+760=3794=3354(이전 전체)-188(구 navi_factory)+628(신규)와 일치)
- Kaggle 업로드 후 이번엔 처음부터 `kaggle datasets status`로 "ready" 확인하고서 커널 push
  (v14 때 겪은 데이터 처리 지연 함정 재발 없음)
- v16 학습(약 65분 소요, imgsz=960이라 이전 640 학습보다 훨씬 오래 걸림) 완료 후 학습 로그에서
  `train: ... 3034 images`, `val: ... 760 images`, `imgsz=960` 재확인(가정하지 않고 항상 로그로
  검증하는 규칙 준수)
- v16 결과: mAP50 0.921 / mAP50-95 0.833 (v15의 0.94/0.87보다 소폭 하락 — 더 어렵고 다양한
  데이터를 학습시킨 자연스러운 트레이드오프, 이전에도 반복된 패턴)
- `test_picture/` 4장을 **imgsz=960으로(학습 해상도와 일치)** 재검증, `frac_w`/`frac_h`까지
  수치로 확인:
  1. 기계/기둥 오탐(Person 0.42) **해결** — 해당 기계가 이제 정확히 PalletJack으로 분류됨
  2. 원거리 사람 미검출 **크게 개선** — 선반 뒤/통로 먼 곳의 여러 사람이 새로 검출됨(이미지 3에서
     5명 이상 신규 검출), 이미지 2에서도 배경의 걷는 사람 신규 검출
  3. 추론 imgsz만 올렸을 때 우려했던 "배경 기둥 전체가 Person으로 잡히는" 새 오탐은 이번엔
     나타나지 않음(전 이미지 frac_w/frac_h 정상 범위)
  4. 기존 정탐(근접 인물 0.94, 지게차 Vehicle 0.80, 이미지 4 사람 검출) 회귀 없이 유지
- 새 가중치는 `data/runs/train-kaggle/weights/`에 반영, v15는 `data/runs/train-kaggle-prev4/`에
  백업(prev3은 v15 이전 백업이라 그대로 유지)
- 앞으로 이 모델로 추론할 때는 `imgsz=960`을 명시해야 학습 해상도와 일치함(`infer.py`의
  `predict()`는 현재 imgsz 인자 없이 기본값 640을 씀 — 다음 세션에서 `infer.py`/`test_gui.py`에
  `imgsz=960` 반영 여부 검토 필요, 이번 세션에서는 검증 스크립트에서만 명시적으로 지정)

## 2026.08.03 (멀티오브젝트 씬 캡처)
- 스케일 문제(ADR-012) 해결 후에도 실제 창고 화면(사람+포크리프트+선반 배경)에서 여전히 탐지
  실패 — "Cluttering" 하나로 뭉뚱그려지고 사람이 아예 안 잡힘. 같은 스케일로 사람/포크리프트를
  단일 오브젝트+깨끗한 배경으로 캡처하면 잘 잡히는 걸 확인(Person 0.82, Vehicle 0.97)해서, 이번엔
  진짜로 "여러 물체 동시에 + 배경 클러터" 자체가 학습 데이터에 없었던 게 원인으로 확인됨(ADR-013)
- `capture_gazebo_dataset.py`에 멀티오브젝트 씬 캡처 추가: 씬 하나에 대상 오브젝트 2~4개(무작위
  클래스, 카메라 시야 안에 각도 슬롯 나눠 배치) + 라벨 없는 배경 클러터(Shelf/Wall) 2~3개를
  같이 스폰. boundingbox_camera는 원래 프레임 안 라벨 붙은 오브젝트 전부를 리포트하는 구조라
  라벨링 자체는 안 바꾸고, "여러 개를 동시에 스폰/배치"하는 월드 구성과 멀티라인 라벨 쓰기만 추가
- `prepare_dataset.py`도 멀티라벨 파일(한 파일에 class_id 여러 줄) 대응 — `_multi_scene` 폴더는
  "첫 줄 class_id"로 그룹핑하면 의미가 없어서, 별도 `multi_scene` 그룹 키로 묶어 그 안에서만
  80/20 분할
- 소규모(2씬) 먼저 테스트해 라벨/이미지가 정확히 나오는지(특히 겹치는 오브젝트끼리 박스가
  안 섞이는지) 눈으로 확인 후, 전체 150씬(카메라 3대, 최대 450장) 캡처 — 407장 확보
- 재학습(2054/513): 전체 mAP는 소폭 하락(mAP50 0.995→0.968, mAP50-95 0.981→0.922, 가려짐 있는
  씬이 섞인 자연스러운 트레이드오프)했지만, held-out 멀티오브젝트 씬 8장 스팟체크에서 정답
  개수·클래스가 예측과 거의 정확히 일치(예: TrashCan+Person×2+PalletJack 4개 전부 검출) — 실제
  목표(여러 물체 동시 인식)는 확실히 달성
- PalletJack의 confidence 편차(0.27~0.94)는 이번에도 여전함(ADR-010 미해결 이슈 지속)

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

