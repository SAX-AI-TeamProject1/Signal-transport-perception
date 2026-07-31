# YOLO 인지 연동 가이드 — 라이다로 트리거되는 물체 분류

## 0. 상태

이 문서는 설계 초안이지 실행 계획이 아니다 — 여기 적힌 것은 아직 아무것도 구현되지 않았다. 두 레포(Signal-Simulation과 이 레포 Signal-transport-perception)가 함께 지향할 형태를 맞추고, 실제 연동을 막는 미해결 지점을 명시하기 위한 문서다.

## 1. 범위와 레포 간 경계

Signal-Simulation의 `CLAUDE.md`에 따라, 카메라 기반 이미지 처리 — YOLO 모델 자체, 학습, 추론까지 전부 — 는 Signal-transport-perception이 전적으로 소유·개발한다. Signal-Simulation은 디텍터를 절대 직접 돌리지 않고, 분류·탐지 코드를 담지 않는다. Signal-Simulation이 담당하는 것은: Gazebo에서 장비에 카메라를 달아주는 것, 그 카메라(그리고 `gazebo-control-guide.md` 7장에서 이미 다룬 라이다)를 ROS 2로 브리지하는 것, 그리고 ROS 2 토픽으로 들어오는 분류 결과에 반응하는 것이다.

시뮬레이션에는 별도의 물리적 "엣지 디바이스"가 없다. Signal-Simulation의 `CLAUDE.md`에 이미 Signal-Vision과 Signal-Simulation이 같은 리눅스 머신에서 돈다고 명시돼 있는데, 여기도 마찬가지다 — 실제 배포라면 장비에 탑재될 "온보드 엣지 추론"이, 시뮬레이션에서는 그저 Signal-Simulation이 발행하는 카메라 토픽을 구독하는 또 다른 ROS 2 노드일 뿐이다. 실제 엣지 하드웨어(예: 실물 로봇에 장착되는 Jetson)는 이후에나 다룰, 지금 범위 밖의 문제다.

## 2. 인지 쪽에 이미 있는 것

`docu/decisions.md`(ADR 형식, 2026-07-21 기준) 내용을 요약한 것이다 — 전체 근거는 그 파일을 직접 읽을 것, 여기는 방향을 잡기 위한 요약일 뿐이다:

- 태스크 유형: 분류가 아니라 객체 탐지(`yolov8n.pt` 기반) — 바운딩박스가 이후 크기 추정에 쓰이기 때문 (ADR-001).
- 학습 이미지는 (Blender도, 실사진도 아닌) Gazebo 자체 카메라 렌더러로 직접 캡처해 도메인 갭을 없앤다 — `capture_gazebo_dataset.py`가 물체를 하나씩 임시 월드에 스폰해서 스윕하며 촬영한다 (ADR-007). `navi_factory.sdf` 자체는 건드리지 않는다.
- 현재 의미 클래스: `person`, 차량군(`car, truck, bus, motorcycle, bicycle, train`), 그리고 창고 사물 4클래스 — `PalletJack, Bucket, Cluttering, TrashCan` (ADR-003, ADR-008). Signal-Simulation의 창고 에셋 중 실제로 존재하는 메시는 `PalletJackB_01`뿐이다 — 지게차나 트럭 메시는 아직 그쪽에 없어서, 매칭되는 Gazebo 에셋이 추가되기 전까지 "지게차/트럭" 클래스는 Gazebo 캡처가 아니라 COCO 사진으로 학습된다.
- 행동 정책(ADR-004, 결정은 됐으나 어디에도 아직 코드화되지 않음): 라이다 범위 안의 `person` → **STOP**, 라이다 범위 안의 차량군 → **AVOID**, 그 외는 추정 실제 크기(바운딩박스 크기 + 라이다 거리)로 판단하며, 무게는 측정할 센서가 없으므로 클래스 기반 휴리스틱으로 근사한다. 크기 임계값과 bbox+거리→크기 공식은 명시적으로 미해결 상태다.
- 레포 경계(ADR-005, Signal-Simulation의 `CLAUDE.md`와 대응): 분류와 STOP/AVOID/통과 판단은 모두 이 레포에서 계산하고, 그 판단 결과만 ROS 2로 Signal-Simulation에 넘어간다. 토픽 이름, 메시지 타입, 좌표계는 양쪽 다 아직 정해지지 않았다.
- 이미 기록돼 있고 Signal-Simulation 쪽에서 바로 처리 가능한 미해결 TODO: `navi_factory`에 사람 모델이 없다 — `person` 클래스의 Gazebo 캡처 학습 데이터도, 이 파이프라인의 실제 엔드투엔드 테스트도 이것 때문에 막혀 있다.

## 3. Signal-Simulation 쪽에 추가되어야 하는 것

### 3.1 장비에 카메라 장착

방법 B 장비 모델(`gazebo-control-guide.md` 6장)에 카메라 센서 링크를 추가한다. 7장에서 라이다 링크를 추가했던 방식 그대로, 차체에 고정된 자식 링크로 붙인다:

```xml
<link name="camera_link">
  <pose relative_to="chassis">0.35 0 0.5 0 0 0</pose>  <!-- 전방을 향하고, 화물칸 위쪽에 위치 -->
  <inertial><mass>0.05</mass></inertial>
  <sensor name="camera" type="camera">
    <topic>/model/transport_1/camera</topic>
    <update_rate>15</update_rate>
    <camera>
      <horizontal_fov>1.396</horizontal_fov>
      <image><width>640</width><height>480</height></image>
      <clip><near>0.1</near><far>15.0</far></clip>
    </camera>
    <always_on>true</always_on>
  </sensor>
</link>
<joint name="camera_joint" type="fixed">
  <parent>chassis</parent>
  <child>camera_link</child>
</joint>
```

이 카메라는 라이다에 쓰던 것과 같은 `gz-sim-sensors-system` / `ogre2` 플러그인을 재사용한다(`gazebo-control-guide.md` 7장) — 일반 `camera` 센서 타입에는 월드 레벨 플러그인을 추가로 넣을 필요가 없다.

6장의 기존 라이다/오도메트리 브리지와 함께 ROS 2로 브리지한다:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /model/transport_1/camera@sensor_msgs/msg/Image@gz.msgs.Image \
  /model/transport_1/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo
```

### 3.2 `navi_factory`에 사람 모델 추가

월드에 사람 모델 또는 actor를 추가한다 — 2장에서 언급한 TODO를 해소하는 작업이며 양쪽 레포 모두에 필요하다. 카메라에 보이기만 하면 되는 목적이라면 (`gazebo-control-guide.md` 2장의) 걷는 방법 A `<actor>`로 충분하다. 다만 actor는 충돌체가 없어서, "카메라+라이다 근접 안의 actor"를 STOP 트리거로 인정하도록 판단 로직을 별도로 만들지 않는 한 라이다 거리 기반 STOP을 스스로 유발하지 않는다 — ADR-004의 STOP 트리거는 라이다가 감지한 물체를 전제로 하므로, 행동 테스트에 이 방식을 쓰기 전에 인지 쪽과 먼저 맞춰볼 것.

(학습 데이터 캡처가 아니라) 실제 STOP 행동 테스트가 목표라면, 라이다가 장애물로 인식하도록 실제 충돌체를 가진 사람 표현이 필요하다 — 순수 actor가 아니라 정적이거나 느리게 움직이는 단순 충돌 모델이 될 수 있다. 이건 실제로 행동 테스트를 만들 때 결정하면 되고, 3.1의 카메라/브리지 작업을 막지는 않는다.

### 3.3 Signal-Simulation과의 인터페이스 계약

어느 쪽도 아직 토픽 이름, 메시지 타입, 좌표계를 확정하지 않았다(ADR-005의 미해결 항목). 아래는 논의를 시작하기 위한 시안이지 결정이 아니다 — 어느 쪽이든 이걸 기준으로 코드를 짜기 전에 Signal-Simulation을 담당하는 쪽과 반드시 확인할 것:

| 항목 | 시안 |
|---|---|
| 토픽 | `/transport_1/perception/action` |
| 메시지 | 커스텀 메시지, 예: `PerceptionAction{string action; string object_class; float32 distance_m; float32 confidence}` (`action`은 `stop`/`avoid`/`pass` 중 하나) — 또는 이 레포가 원본 탐지와 판단 결과를 둘 다 발행하고 싶다면, 원본 탐지용으로 `vision_msgs/Detection2DArray`를 재사용하고 이미 판단된 액션은 별도 토픽으로 |
| 좌표계 | 장비 기준 상대 좌표(`transport_1/base_link` 등) — stop/avoid/pass 판단 자체가 이 장비의 다음 동작에 관한 것이지 월드 좌표계 위치가 아니기 때문 |

Signal-Simulation은 STOP/AVOID/pass 판단 로직(크기 임계값, bbox+거리 공식)을 직접 만들어서는 안 된다 — ADR-005의 경계상 그건 이 레포(Signal-transport-perception) 소관이다. Signal-Simulation은 위의 '결과' 스키마만 알면 된다.

### 3.4 판단 결과를 소비하는 노드 (Signal-Simulation 쪽)

Signal-Simulation의 작은 ROS 2 노드가 3.3의 인지-액션 토픽과 `/model/transport_1/odometry`를 구독하고, 웨이포인트 팔로워가 이미 발행하는 것과 같은 `cmd_vel`을 구동한다(`gazebo-control-guide.md` 6장). 이는 수신호 명령에 대해 이미 설명된 것과 같은 오버라이드 지점이다: "장비가 수신호 포인트에 도착하면 팔로워는 정지하고 정지/좌회전/우회전 명령에 제어권을 넘긴다" — 인지 기반 STOP/AVOID도 같은 종류의 '일시정지 후 제어권 이양'이며, 다만 수신호 포인트가 아니라 경로 중간 어디서든 트리거된다는 점만 다르다.

실제 정책은 이 레포(Signal-transport-perception)가 발행하는 값에 맡기고, Signal-Simulation 쪽 동작은 다음과 같이 제안한다:

- `action == "stop"` → `cmd_vel`을 0으로 만들고, 인지 토픽이 장애물 해소를 알릴 때까지 유지.
- `action == "avoid"` → 웨이포인트 팔로워를 일시정지하고 대기하거나 간단한 측면 회피 기동을 수행한다. 완전한 경로 재계획은 Phase 2 장애물 회피 영역(`CLAUDE.md` Phase 2)이며, 첫 구현에서는 필요하지 않다.
- `action == "pass"` → 아무 것도 하지 않고 웨이포인트 팔로워가 계속 주행.

## 4. Signal-Simulation 쪽 권장 패키지 구조 (`gazebo-control-guide.md` 9장 확장)

```
src/signal_transport/
├── config/
│   └── tracks.yaml
├── worlds/
│   └── warehouse_tracks.sdf        # 이제 3.1의 camera_link도 포함
├── models/
│   ├── transport_vehicle/          # + camera_link, camera_joint
│   └── person/                     # 3.2 — 새로 추가
├── launch/
│   └── sim.launch.py               # 3.1의 카메라 브리지 추가
└── signal_transport/
    ├── waypoint_follower.py
    └── perception_action_node.py   # 3.4 — 새로 추가
```

## 5. 단계 구분

이 작업은 Phase 2(Signal-Simulation의 `CLAUDE.md` 기준, Phase 1 수신호 제어 위에 얹히는 장애물 감지)다 — `gazebo-control-guide.md`에 이미 설명된 수신호 제어 경로를 막거나 대체해서는 안 된다. 카메라+브리지(3.1)는 독립적이고 이 레포 자체 테스트도 풀어주므로 먼저 만들고, 판단 노드(3.4)는 인터페이스 계약(3.3)이 실제로 합의된 뒤로 미뤄도 된다.

## 6. 실제 연동 전에 풀어야 할 미해결 질문들

- 인지→시뮬레이션 인터페이스의 토픽 이름, 메시지 타입, 좌표계 (3.3) — 어느 한쪽에서 일방적으로 정하지 말고 두 레포가 합의해야 함.
- 충돌체가 없는 방법 A 사람 actor로 행동 테스트가 충분한지, 아니면 충돌 가능한 모델이 필요한지 (3.2).
- 크기 임계값과 bbox+거리→크기 공식이 아직 미해결(ADR-004) — 그게 정해지기 전까지 Signal-Simulation의 판단 노드는 pass 로직을 확정할 수 없음.
- Signal-Simulation의 에셋에 아직 지게차/트럭 메시가 없음 (2장) — "지게차나 트럭"이 COCO로만 학습된 클래스명이 아니라 `navi_factory` 안에서 실제로 카메라에 보이는 물리적 장애물로도 존재해야 한다면 필요.
