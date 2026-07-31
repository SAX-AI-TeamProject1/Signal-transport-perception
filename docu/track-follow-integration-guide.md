# 트랙 추종 연동 가이드 — 카메라로 바닥 트랙을 보고 코너링까지 하는 자율주행

## 0. 상태

이 문서는 설계 초안이지 실행 계획이 아니다 — 인지(비전) 쪽 함수는 이 레포에 구현·검증까지 끝났지만,
Signal-Simulation 쪽 연동(실제 조향에 반영하는 부분)은 아직 아무것도 구현되지 않았다. 두 레포가
함께 지향할 형태를 맞추고, 실제 연동을 막는 미해결 지점을 명시하기 위한 문서다.

`docu/yolo-perception-integration-guide.md`(라이다로 트리거되는 STOP/AVOID 장애물 분류용
인계 문서)와는 별개의 기능이다 — 그쪽은 "충돌 위험 시 물체가 뭔지 판단", 이쪽은 "상시 카메라로
바닥 트랙을 보고 조향을 보정"으로, 트리거 방식도 소비하는 쪽도 다르다. 다만 이쪽도 같은 YOLO
모델(`data/runs/train-kaggle/weights/best.pt`)의 탐지 결과를 입력으로 재사용한다(2.2 참고).

## 1. 배경 — 왜 필요한가

navi_factory 월드 바닥에는 로봇이 시각적으로 따라가야 할 주황색 대시라인 트랙(`track_*` 모델들:
`track_ne_entry_*`, `track_nw_entry_*`, `track_signal_entry_*`, `track_ne_dash_*` 등)이 깔려
있고, 직선 구간뿐 아니라 분기점·반환점·목표지점(원형 패드, 콘+원형 마커)까지 있는 실제 라우팅
네트워크다.

Signal-Simulation에는 이미 이 트랙을 도는 patrol 기능이 있다(`auto-patrol` 브랜치,
`config/tracks.yaml` + `src/robot_control/robot_control/patrol/waypoint_follower.py`).
**다만 현재 이 patrol은 카메라가 아니라 `pose_gt`(시뮬레이션 그라운드트루스 위치)로 조향한다** —
자기 위치를 완벽히 아는 상태로 미리 정해진 좌표를 그대로 따라가는 것이라, 실제 로봇이라면 있을 수
없는 방식이고 "카메라로 보고 판단"하는 자율주행이 아니다.

목표는 사람 운전에 빗대면: `tracks.yaml`(내비게이션, 어디로 가야 하는지 큰 그림)은 그대로 두고,
**지금 이 순간 핸들을 얼마나 꺾을지는 카메라로 실제 보이는 트랙 위치가 결정**하도록 바꾸는 것 —
Tesla류 자율주행이 지도/GPS(매크로 경로)와 카메라 차선 인식(실시간 로컬 보정)을 분리해서 쓰는 것과
같은 구조. 분기 선택·반환·목표지점 판단 같은 상위 라우팅은 여전히 `tracks.yaml`의 몫으로 남긴다
(ADR-011).

## 2. 인지 쪽에 이미 있는 것 (이 레포, 구현·검증 완료)

### 2.1 `src/track_follow.py` — 트랙 인식 함수

```python
def estimate_track(
    frame_bgr: np.ndarray,
    obstacle_boxes: list[tuple] | None = None,
    previous: TrackEstimate | None = None,
) -> TrackEstimate | None:  # (offset, curvature, confidence) 또는 트랙 못 찾으면 None
```

- **offset**: 화면 하단(로봇 바로 앞)에서 트랙 중심이 화면 중앙 대비 얼마나 치우쳤는지, `-1`(왼쪽
  끝)~`1`(오른쪽 끝) 정규화값.
- **curvature**: 가까운 밴드~먼 밴드로 나눠 피팅한 2차 다항식의 2차항 — 0에 가까우면 직진, 부호로
  꺾이는 방향을 안다.
- **confidence**: 이번 프레임에서 트랙을 얼마나 확실하게 찾았는지(0~1). 다른 차량에 가려 일부만
  보이면 낮아진다.
- **obstacle_boxes**: 이 레포의 YOLO 탐지 결과(Vehicle/Person 등 바운딩박스, `(x1,y1,x2,y2)`
  리스트)를 그대로 넘기면, 그 영역을 트랙 색상 마스크에서 제외한다 — 다른 차량이 트랙을 가리거나
  헷갈리게 하는 문제 대응(2.3 참고).
- **previous**: 직전 프레임의 추정치를 넘기면 (1) 오프셋이 갑자기 크게 튀는 걸 억제(다른 분기로
  잘못 물린 것으로 간주)하고 (2) 프레임 간 값을 평활화한다. **호출부가 프레임마다 이 상태를 들고
  있어야 한다** — `estimate_track` 자체는 상태를 저장하지 않는 순수 함수.
- **`select_own_track`**: 화면에 여러 트랙 분기가 동시에 보여도, 로봇 바로 앞(화면 하단 중앙)에
  맞닿은 connected-component 하나만 "지금 타고 있는 트랙"으로 선택 — 무관한 다른 차선/분기는
  무시한다. 분기점에서 "어느 쪽으로 갈지"를 판단하는 게 아니라, 여러 개가 보일 때 헷갈리지 않게
  거르는 역할만 한다는 점에 유의.

이 함수는 **분기 선택·반환·목표지점 도착 같은 라우팅 판단을 하지 않는다.** 그건 여전히
`tracks.yaml`/`waypoint_follower.py`의 책임이다(ADR-011).

### 2.2 YOLO 장애물 모델 (obstacle_boxes 공급원)

6클래스(`PalletJack, Bucket, Cluttering, TrashCan, Person, Vehicle`), 전체 mAP50 0.985 /
mAP50-95 0.955. `data/`는 git에 올라가지 않으므로(재현 가능 산출물, ADR-006) 모델 자체가
git으로는 전달되지 않는다 — 대신 `src/weights.py`의 `resolve_weights()`가 (1) 이 레포 루트에서
실행 중이면 `data/runs/train-kaggle/weights/best.pt`를, (2) 다른 레포에서 이 패키지를
import해서 쓰는 경우(CWD가 다름)에는 `~/.cache/signal-transport-perception/`를, (3) 거기에도
없으면 Kaggle 커널(`heojaeseong/yolo-percep`)에서 자동으로 받아온다. `infer.py`의
`load_model()`이 이걸 그대로 쓰므로, Signal-Simulation 쪽은 **Kaggle 계정 접근 권한만 있으면
이 패키지를 import하는 순간 모델이 자동으로 준비된다** — `/tmp`(레포 밖)에서 직접 실행해
end-to-end로 검증 완료. `Vehicle`(Forklift)과 `Person` 탐지 결과를
2.1의 `obstacle_boxes`로 넘기면 된다. `PalletJack`만 mAP50-95 0.838로 다소 처지고 학습 스윕 밖
각도에서 중복 박스가 관찰됐다(decisions.md ADR-010 미해결 항목) — 트랙 마스킹 용도로는 영향이
크지 않을 것으로 보이나(PalletJack이 트랙 위에 있을 일이 적음) 참고할 것.

### 2.3 검증 내역

- HSV 색상 임계값은 실제 Gazebo(ogre2) 렌더로 `track_ne_entry_1`을 스폰해 캡처한 프레임에서 픽셀을
  직접 샘플링해 캘리브레이션함 — 그림자 진 면도 색상(H)·채도(S)는 유지되고 명도(V)만 낮아짐을 확인.
- 합성 이미지로 스트레스 테스트: 트랙에 바로 맞닿아 겹치는 가짜 오브젝트가 있을 때, 마스킹 없이는
  `offset=0.20`(틀림)+`confidence=1.0`(거짓 확신)으로 나오다가 `obstacle_boxes`로 마스킹하면
  `offset≈0`(정답)+`confidence=0.46`(더 정직한 값)으로 정상화됨을 확인.
- **아직 안 한 것**: 실제 움직이는 로봇의 라이브 카메라 피드로 검증한 적이 없다 — 지금까지는 정적
  Gazebo 캡처 + 합성 이미지로만 검증했다. 실제 주행 중 프레임(모션 블러, 카메라 진동, 실시간 처리
  지연 등)에서 얼마나 잘 동작하는지는 미검증.

## 3. Signal-Simulation 쪽에 추가되어야 하는 것

### 3.1 카메라 프레임을 어디서 처리할지 — 아직 미정, 권장안

`docu/yolo-perception-integration-guide.md` 3.3의 기존 경계 원칙(카메라 기반 이미지 처리는 전부
이 레포 소관, Signal-Simulation은 결과만 소비)을 그대로 따르면:

- 이 레포(또는 이 레포 코드를 감싼 별도 ROS 2 노드)가 Signal-Simulation이 발행하는 카메라 토픽을
  구독하고, YOLO 추론(2.2) → `estimate_track`(2.1) 순으로 돌려서 `(offset, curvature,
  confidence)`를 ROS 2 토픽으로 발행한다.
- Signal-Simulation은 그 토픽만 구독하면 되고, 비전 처리 코드를 직접 갖지 않는다.

**다만 이건 시안이지 결정이 아니다** — 이전 인계 문서와 마찬가지로 실제 코드를 짜기 전에
Signal-Simulation 쪽과 반드시 확인할 것. 특히 이번 건은 라이다 트리거 방식과 달리 **매 프레임(예:
15Hz) 상시 실행**돼야 해서, 별도 프로세스 간 ROS 2 통신 오버헤드가 조향 반응 속도에 영향을 줄 수
있다 — 지연이 문제가 되면 Signal-Simulation 쪽에서 이 레포를 라이브러리로 직접 import해서 같은
프로세스 안에서 도는 방안도 검토 대상(다만 이건 ADR-005의 레포 경계 원칙과 긴장 관계이므로 가볍게
결정하지 말 것).

### 3.2 `waypoint_follower.py` 조향 로직에 블렌딩

현재 `waypoint_follower.py`는 `pose_gt` 기반 헤딩 오차만으로 `cmd_vel_auto`를 계산한다. 여기에
2.1의 `(offset, curvature, confidence)`를 섞어서, "웨이포인트가 주는 큰 방향"은 유지하되 "순간
조향각은 실제 보이는 트랙이 지배"하도록 바꿔야 한다. 구체적 블렌딩 공식(가중치, `confidence`가
낮을 때 `pose_gt`로 얼마나 폴백할지 등)은 여기서 결정하지 않았다 — Signal-Simulation 쪽 조향
튜닝 경험이 있는 사람이 정하는 게 맞다고 보고 미해결로 남겨둔다.

### 3.3 인터페이스 계약 (안)

| 항목 | 시안 |
|---|---|
| 토픽 | `/transport_1/perception/track_estimate` |
| 메시지 | 커스텀 메시지, 예: `TrackEstimate{float32 offset; float32 curvature; float32 confidence; bool track_visible}` — `track_visible=false`면 `estimate_track`이 `None`을 반환한 경우(트랙을 아예 못 찾음), 이때 `waypoint_follower.py`는 `pose_gt` 폴백으로 완전히 전환해야 함 |
| 갱신 주기 | 카메라 프레임레이트에 맞춤(예: 15Hz) — 라이다 트리거 방식(이벤트성)과 다름 |

## 4. 패키지 구조 제안 (Signal-Simulation 쪽)

```
src/robot_control/robot_control/patrol/
├── waypoint_follower.py       # 기존 — pose_gt 헤딩 계산에 3.3 토픽 블렌딩 추가
└── track_estimate_bridge.py   # 신규(3.1) — 카메라 구독 → (이 레포 호출) → 3.3 토픽 발행
```

## 5. 미해결 질문들

- 카메라 프레임 처리를 어느 프로세스/레포에서 돌릴지(3.1) — 레포 경계 vs 지연시간 트레이드오프,
  두 레포가 합의 필요.
- 3.3의 토픽 이름/메시지 타입/갱신 주기 — 어느 한쪽에서 일방적으로 정하지 말 것.
- `pose_gt` 폴백과 비전 보정 사이의 블렌딩 가중치, `confidence` 임계값(3.2) — 조향 튜닝 영역.
- 라이브(실제 주행 중) 검증이 전혀 안 된 상태(2.3) — 정적 캡처/합성 이미지 검증만으로 실제 주행에
  바로 투입하지 말고, 먼저 저속·짧은 구간에서 라이브 테스트할 것을 권장.
- `tracks.yaml`은 현재 `ne_entry` 트랙 계열 하나만 웨이포인트로 정의돼 있고, 스크린샷에 보였던
  다른 `track_*` 계열(`nw_entry`, `signal_entry`, `ne_dash` 등)은 아직 라우팅에 안 들어가 있음 —
  이 레포가 아니라 Signal-Simulation 쪽에서 확장할 부분.

## 6. 참고 — 기존 인계 문서(`yolo-perception-integration-guide.md`)의 최신화 필요 항목

그 문서 작성 시점(2026-07-21 ADR 기준) 이후 다음 항목들이 실제로 해소됐다 — 다음에 그 문서를 다시
볼 사람이 헷갈리지 않도록 남겨둔다:

- 2장 "navi_factory에 사람 모델이 없다" → **해소됨**. `StandingPerson/WalkingPerson/CasualFemale`
  모델이 이미 배치돼 있고, `Person` 클래스로 학습까지 완료(mAP50-95 0.992).
  `docu/decisions.md` ADR-010 참고.
- 2장·6장 "지게차/트럭 메시가 없어 COCO로만 학습" → **해소됨**. `Forklift` 메시가 배치돼 있고
  `Vehicle` 클래스로 Gazebo 네이티브 캡처·학습 완료(mAP50-95 0.981).
- 클래스 구성이 4개(`PalletJack, Bucket, Cluttering, TrashCan`)에서 **6개**(+ `Person`,
  `Vehicle`)로 확장됨. 라벨링 방식도 수동 half-extent 투영에서 gz-sim `boundingbox_camera` 자동
  라벨링으로 전면 교체됨(ADR-010).
