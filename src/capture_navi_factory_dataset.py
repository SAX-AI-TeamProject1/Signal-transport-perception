# Last updated: 2026-08-04
'''
navi_factory.sdf(Signal-Simulation의 실제 시뮬레이션 월드)를 그대로 띄우고, 그 안을 실제 로봇
카메라와 동일한 스펙의 "캡처 리그"가 트랙을 따라 돌아다니며 촬영한다.

배경: capture_gazebo_dataset.py로 만든 합성 데이터로 두 번 재학습해도(배경 구조물 오탐지 수정,
전체 프레임 라벨 버그 수정) 실제 배포 환경(navi_factory)에서 Person/PalletJack 오탐지가 여전히
남아 있었다 — 합성 월드가 실제 배포 환경의 조명/배경/스케일을 완벽히 재현하지 못하는 domain gap이
근본 원인으로 보여, 아예 실제 navi_factory.sdf에서 직접 캡처하기로 방향을 바꿨다.

navi_factory.sdf는 이 프로젝트 소유가 아니라 Signal-Simulation의 유일한 실제 시뮬레이션 월드라서
절대 읽기 전용으로만 다룬다(수정 금지 — capture_gazebo_dataset.py의 ADR-007과 동일 원칙). 그래서:
  - 로봇을 실제로 주행시키는 대신(ROS2/물리 스택 불필요), Signal-Simulation의
    config/tracks.yaml에 있는 파견 경로 좌표(zone_ne/zone_nw routes)를 그대로 읽어서 카메라
    리그를 set_pose 서비스로 그 위를 순간이동시킨다.
  - navi_factory.sdf에는 boundingbox_camera 라벨 플러그인이 안 붙어 있고(우리가 스폰한 게
    아니라서), 라벨을 자동으로 못 얻는다 — 대신 각 오브젝트의 알려진 pose + 실측 half-extent를
    핀홀 카메라 모델로 직접 투영해서 라벨을 계산한다. 정적 오브젝트(PalletJack 2개,
    hand_signal_person)는 navi_factory.sdf에 박힌 고정 pose를 그대로 쓰고, 걸어다니는 Person
    액터 8개는 navi_factory.sdf의 <trajectory> 웨이포인트를 그대로 읽어 sim_time으로 선형보간한다
    (gz-sim이 내부적으로 하는 것과 동일한 계산 — dynamic_pose/info 토픽이 이 액터들의 pose를
    브로드캐스트하지 않는 걸 실측으로 확인해서, 액터를 건드리지 않고 우리가 직접 계산하는 쪽으로
    결정함).
  - 라벨이 하나도 안 보이는 프레임도 버리지 않고 그대로 저장한다(빈 라벨 파일) — 이번 캡처의
    목적 자체가 "실제 배포 환경에서 오탐지를 줄이는 것"이라, 실제 배경(선반/MecanumLift
    소품/기둥 등)만 있는 프레임이 오히려 가장 값진 hard negative다.

1차 캡처(188장) 재학습 후 검증에서 두 문제가 남음: (1) 벽/기둥/콘솔 등 구조물이 화면을 가득
채우면 Person으로 오탐, (2) 멀리 있는 사람이 미검출. 원인은 전방만 보고 지나쳐서 구조물
클로즈업 hard negative가 부족했고, 트랙을 한 바퀴만 돌아서 액터와의 거리 조합 다양성도
부족했던 것으로 추정 — SIDE_GLANCE_EVERY/SIDE_GLANCE_YAWS(좌우 글랜스 프레임 추가)와
NUM_PASSES(트랙을 여러 번 돌아 다른 sim_time=다른 액터 위치 확보)로 보강함.

카메라 스펙은 실제 로봇(mecanum_lift_robot.urdf.xacro의 sim_camera)과 동일하게 맞췄다:
horizontal_fov=1.047, 640x480, 섀시 로컬 오프셋 (1.0, 0, 1.0). 로봇 스폰 z(0.3, bringup.launch.py의
robot_info)를 더해 카메라 절대 높이 1.3m로 계산한다. 라벨 투영에 쓰는 오브젝트 half-extent는
스크래치패드에서 따로 실측한 값(measure_palletjack_axes.py)과 navi_factory.sdf 자체의 authoritative
collision 정의(hand_signal_person)를 하드코딩해 가져왔다 — 계측 과정 자체는 이 스크립트에 없다.

.venv-gz/bin/python src/capture_navi_factory_dataset.py
'''

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_stats_pb2 import WorldStatistics
from gz.transport13 import Node
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REPO_ROOT.parent / "Signal-Simulation"
NAVI_FACTORY_SDF = SIM_ROOT / "worlds" / "navi_factory" / "world" / "navi_factory" / "navi_factory.sdf"
SIM_MODELS_DIR = SIM_ROOT / "worlds" / "navi_factory" / "models"
TRACKS_YAML = SIM_ROOT / "config" / "tracks.yaml"

OUT_DIR = REPO_ROOT / "data" / "raw" / "gazebo_capture" / "_navi_factory"
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"

WORLD_NAME = "warehouse_fuel"  # navi_factory.sdf 자체의 <world name> (수정 금지, 읽기 전용 참조만)
RIG_NAME = "capture_rig"
RIG_CAMERA_TOPIC = "capture_rig_cam"

SEMANTIC_CLASSES = ["PalletJack", "Bucket", "Cluttering", "TrashCan", "Person", "Vehicle"]

# 실제 로봇 카메라 스펙(mecanum_lift_robot.urdf.xacro의 sim_camera) 그대로 재현 — capture_gazebo_dataset.py의
# 정사각형(640x640) 합성 캡처와 달리 실제 배포 화면 비율(640x480)을 그대로 쓴다.
IMG_W, IMG_H = 640, 480
HFOV = 1.047
VFOV = 2 * math.atan(math.tan(HFOV / 2) * (IMG_H / IMG_W))
NEAR, FAR = 0.1, 100.0

# 로봇 스폰 z(bringup.launch.py의 robot_info, 0.3=섀시 지면 높이) + 카메라 섀시 로컬 오프셋(1.0,0,1.0).
CHASSIS_Z = 0.3
CAMERA_FORWARD_OFFSET = 1.0  # 카메라가 섀시 원점보다 앞으로 나온 거리(로컬 x)
CAMERA_Z = CHASSIS_Z + 1.0

MIN_BOX_PIXELS = 4
STEP_DISTANCE = 1.0  # 트랙 리샘플링 간격(m) — 캡처 지점 밀도

# 재검증(2026-08-04)에서 남은 두 가지 문제에 대응해 캡처 밀도/다양성을 늘린다:
#   1) 벽/기둥/콘솔 같은 배경 구조물이 화면을 가득 채우는 "Person" 오탐 — 지금까지는 전방만
#      보고 지나쳤을 뿐, 옆을 클로즈업으로 본 프레임이 학습 데이터에 없었다. SIDE_GLANCE_EVERY
#      번째 지점마다 좌우로 고개를 돌린 프레임을 추가로 찍어 이런 구조물이 화면을 가득 채운
#      hard negative를 늘린다.
#   2) 멀리 있는 사람 미검출(recall 부족) — 액터는 계속 루프를 도는데 카메라가 트랙을 한 바퀴
#      도는 시간(sim_time)과 액터 주기가 안 맞아떨어져서, 매 패스마다 각 지점에서 액터와의
#      거리 조합이 달라진다. NUM_PASSES로 트랙을 여러 번 돌아 같은 지점에서도 다른 sim_time
#      (=다른 액터 위치/거리)의 프레임을 추가로 확보한다.
SIDE_GLANCE_EVERY = 5
SIDE_GLANCE_YAWS = (math.pi / 2, -math.pi / 2)  # 진행 방향 기준 좌/우 90도
NUM_PASSES = 2

# ---------------------------------------------------------------------------
# 라벨 대상 오브젝트: navi_factory.sdf에 실제로 박혀 있는 pose + 실측/authoritative half-extent.
# ---------------------------------------------------------------------------

# navi_factory.sdf의 hand_signal_person 자체 collision box(<pose>0 0 0.9...</pose>
# <box><size>0.5 0.35 1.8</size></box>)를 모든 Person(정적 1개 + 걷는 액터 8개) 공통으로 쓴다.
# half_x(폭), half_y(두께), half_z(반높이).
PERSON_HALF = (0.25, 0.175, 0.9)

# 스크래치패드 계측(measure_palletjack_axes.py, yaw=0/90도로 축 분리 실측) 결과에 navi_factory
# 실제 배치 스케일(<scale>1.5 1.5 1.5</scale>)을 곱함. half_x=길이, half_y=폭, half_z=반높이(양쪽
# 각도 측정 평균).
PALLETJACK_HALF = (0.5883 * 1.5, 0.2715 * 1.5, (0.5294 + 0.4797) / 2 * 1.5)

# (class, x, y, z, yaw, half_extent) — z는 오브젝트 중심의 월드 높이. PalletJack 둘 다 지면(z=0)에
# 놓인 static 모델이라 중심 높이는 half_z와 같다(원점이 바닥에 닿아 있다고 가정).
STATIC_OBJECTS = [
    ("PalletJack", -9.40929, 20.4472, PALLETJACK_HALF[2], 0.860431, PALLETJACK_HALF),
    ("PalletJack", 14.1462, -22.9096, PALLETJACK_HALF[2], 0.0, PALLETJACK_HALF),
    # hand_signal_person: model pose z(0.2497) + collision 로컬 z 오프셋(0.9) = 실제 world 중심.
    ("Person", 0.0, 40.0, 0.2497 + 0.9, 0.0, PERSON_HALF),
]

# 걷는 Person 액터 8개 — navi_factory.sdf의 <trajectory> 웨이포인트를 그대로 옮겨 적음.
# {actor_name: (delay_start, [(time, x, y, yaw), ...])}. z는 스켈레톤 원점 관례가 메시마다 달라서
# (MaleVisitor는 엉덩이 높이 ~1.05, FemaleVisitorWalk는 발밑 ~0.09) 신뢰하지 않는다 — 대신 "사람은
# 바닥에 서 있고 키는 hand_signal_person과 같다"고 가정해 중심 높이를 PERSON_HALF[2]로 고정하고,
# 웨이포인트에서는 x, y, yaw만 보간해서 쓴다.
ACTOR_TRAJECTORIES = {
    "path_prop_2_person": (0.0, [
        (0.0, 3.495, 9.0, 0.0), (8.125, 9.995, 9.0, 0.0), (9.125, 9.995, 9.0, 3.142),
        (17.25, 3.495, 9.0, 3.142), (18.25, 3.495, 9.0, 0.0),
    ]),
    "path_prop_3_person": (0.6, [
        (0.0, -3.704, 9.876, 0.877), (9.375, 1.092, 15.642, 0.877), (10.375, 1.092, 15.642, -2.265),
        (19.75, -3.704, 9.876, -2.265), (20.75, -3.704, 9.876, 0.877),
    ]),
    "path_prop_6_person": (1.2, [
        (0.0, -3.14, 5.283, 0.178), (8.125, 3.257, 6.434, 0.178), (9.125, 3.257, 6.434, -2.964),
        (17.25, -3.14, 5.283, -2.964), (18.25, -3.14, 5.283, 0.178),
    ]),
    "path_prop_8_person": (0.0, [
        (0.0, 3.577, 15.642, 4.996), (7.5, 5.256, 9.882, 4.996), (8.5, 5.256, 9.882, 1.854),
        (16.0, 3.577, 15.642, 1.854), (17.0, 3.577, 15.642, 4.996),
    ]),
    "path_prop_17_person": (1.8, [
        (0.0, 3.867, -8.136, 5.994), (3.75, 6.742, -8.992, 5.994), (4.75, 6.742, -8.992, 2.852),
        (8.5, 3.867, -8.136, 2.852), (9.5, 3.867, -8.136, 5.994),
    ]),
    "path_prop_20_person": (2.4, [
        (0.0, -3.061, -14.826, 4.045), (9.375, -7.703, -20.717, 4.045), (10.375, -7.703, -20.717, 0.903),
        (19.75, -3.061, -14.826, 0.903), (20.75, -3.061, -14.826, 4.045),
    ]),
    "path_prop_21_person": (3.0, [
        (0.0, 0.765, -15.126, 5.351), (9.375, 5.236, -21.148, 5.351), (10.375, 5.236, -21.148, 2.209),
        (19.75, 0.765, -15.126, 2.209), (20.75, 0.765, -15.126, 5.351),
    ]),
    "path_prop_23_person": (3.6, [
        (0.0, -4.187, -8.045, 1.616), (8.75, -4.503, -1.052, 1.616), (9.75, -4.503, -1.052, -1.526),
        (18.5, -4.187, -8.045, -1.526), (19.5, -4.187, -8.045, 1.616),
    ]),
}


# 두 각도 사이를 최단 방향으로 선형보간한다(예: 3.0 -> -3.1처럼 경계를 넘는 웨이포인트 대비).
def _lerp_angle(a: float, b: float, t: float) -> float:
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t


# 액터 하나의 현재 (x, y, yaw)를 트래젝토리 웨이포인트 + sim_time으로 계산한다 — gz-sim이 내부적으로
# 액터 애니메이션에 쓰는 것과 동일한 "구간별 선형보간 + loop" 계산을 바깥에서 재현한다.
def actor_pose_at(name: str, sim_time: float) -> tuple[float, float, float]:
    delay, waypoints = ACTOR_TRAJECTORIES[name]
    duration = waypoints[-1][0]
    t = (sim_time - delay) % duration
    for (t0, x0, y0, yaw0), (t1, x1, y1, yaw1) in zip(waypoints, waypoints[1:]):
        if t0 <= t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            x = x0 + (x1 - x0) * frac
            y = y0 + (y1 - y0) * frac
            yaw = _lerp_angle(yaw0, yaw1, frac)
            return x, y, yaw
    _, x, y, yaw = waypoints[-1]  # 부동소수 경계 대비 폴백
    return x, y, yaw


# tracks.yaml의 zone_ne/zone_nw 파견 경로 waypoint를 이어붙인다 — 둘 다 signal_point에서 시작해
# signal_point로 끝나서 그대로 이으면 자연스럽게 연결된다(끝점=다음 시작점).
def load_route_waypoints() -> list[tuple[float, float]]:
    data = yaml.safe_load(TRACKS_YAML.read_text())
    points = []
    for route_name in ("zone_ne", "zone_nw"):
        points += [(p[0], p[1]) for p in data["routes"][route_name]["waypoints"]]
    return points


# 원래 waypoint 사이를 STEP_DISTANCE 간격으로 잘게 나누고, 각 지점에 다음 지점을 향한 접선
# heading을 붙인다(코너는 원본 데이터가 이미 3점 호로 근사해뒀으니 추가 스무딩 없이 직선 보간만).
def resample_path(waypoints: list[tuple[float, float]], step: float) -> list[tuple[float, float, float]]:
    dense = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-6:
            continue
        heading = math.atan2(y1 - y0, x1 - x0)
        n_steps = max(1, round(seg_len / step))
        for i in range(n_steps):
            frac = i / n_steps
            dense.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, heading))
    return dense


# 트랙 위 (x, y, heading) 한 점을 실제 카메라의 월드 pose로 바꾼다: 섀시가 heading 방향을 보고
# 있다고 가정하고, 카메라의 섀시-로컬 전방 오프셋(1.0)만큼 heading 방향으로 밀어낸다.
def camera_pose_at_track_point(x: float, y: float, heading: float) -> tuple[float, float, float, float]:
    cam_x = x + math.cos(heading) * CAMERA_FORWARD_OFFSET
    cam_y = y + math.sin(heading) * CAMERA_FORWARD_OFFSET
    return cam_x, cam_y, CAMERA_Z, heading


# 오브젝트의 3D 오리엔티드 박스(center + yaw 회전 + half_extent) 8개 꼭짓점을 카메라 pose로 핀홀
# 투영해서 화면상 AABB(YOLO 정규화 좌표)를 계산한다. 카메라는 롤/피치 없이 수평(camera_joint의
# rpy 0 0 0과 동일 가정)이라고 본다 — forward=카메라 정면축, right=화면 오른쪽, up=화면 위쪽.
# 가려짐(occlusion)은 계산하지 않는다(알려진 한계 — 파일럿 캡처에서 시각 검증으로 크게 어긋나는
# 경우가 있는지 먼저 확인한다).
def project_box(cam_pose: tuple, center: tuple, half_extent: tuple, obj_yaw: float) -> tuple | None:
    cam_x, cam_y, cam_z, cam_yaw = cam_pose
    forward = (math.cos(cam_yaw), math.sin(cam_yaw))
    right = (math.sin(cam_yaw), -math.cos(cam_yaw))
    hx, hy, hz = half_extent
    cx, cy, cz = center
    cos_o, sin_o = math.cos(obj_yaw), math.sin(obj_yaw)

    corners = []
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            for sz in (-hz, hz):
                wx = cx + sx * cos_o - sy * sin_o
                wy = cy + sx * sin_o + sy * cos_o
                wz = cz + sz
                corners.append((wx, wy, wz))

    tan_h = math.tan(HFOV / 2)
    tan_v = math.tan(VFOV / 2)
    xs, ys = [], []
    for wx, wy, wz in corners:
        rx, ry, rz = wx - cam_x, wy - cam_y, wz - cam_z
        xf = rx * forward[0] + ry * forward[1]
        if xf < NEAR:
            continue
        yr = rx * right[0] + ry * right[1]
        u = 0.5 + 0.5 * (yr / (xf * tan_h))
        v = 0.5 - 0.5 * (rz / (xf * tan_v))
        xs.append(u * IMG_W)
        ys.append(v * IMG_H)

    if len(xs) < 2:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max < 0 or x_min > IMG_W or y_max < 0 or y_min > IMG_H:
        return None
    x_min, x_max = max(0.0, x_min), min(float(IMG_W), x_max)
    y_min, y_max = max(0.0, y_min), min(float(IMG_H), y_max)
    if x_max - x_min < MIN_BOX_PIXELS or y_max - y_min < MIN_BOX_PIXELS:
        return None
    box_cx, box_cy = (x_min + x_max) / 2 / IMG_W, (y_min + y_max) / 2 / IMG_H
    box_w, box_h = (x_max - x_min) / IMG_W, (y_max - y_min) / IMG_H
    return box_cx, box_cy, box_w, box_h


# 캡처 리그(카메라 센서 하나짜리 static 모델) SDF — navi_factory.sdf를 전혀 건드리지 않고
# /world/{world}/create 서비스로 런타임에 스폰한다.
def rig_sdf() -> str:
    return f"""<sdf version="1.6">
  <model name="{RIG_NAME}">
    <static>true</static>
    <pose>0 0 {CAMERA_Z} 0 0 0</pose>
    <link name="link">
      <sensor name="cam" type="camera">
        <camera>
          <horizontal_fov>{HFOV}</horizontal_fov>
          <image><width>{IMG_W}</width><height>{IMG_H}</height></image>
          <clip><near>{NEAR}</near><far>{FAR}</far></clip>
        </camera>
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <visualize>false</visualize>
        <topic>{RIG_CAMERA_TOPIC}</topic>
      </sensor>
    </link>
  </model>
</sdf>"""


def spawn_rig(node: Node) -> None:
    req = EntityFactory()
    req.sdf = rig_sdf()
    req.name = RIG_NAME
    ok, res = node.request(f"/world/{WORLD_NAME}/create", req, EntityFactory, Boolean, 3000)
    if not ok or not res.data:
        raise RuntimeError("capture rig 스폰 실패 (/world/.../create)")


# set_pose 서비스로 이름이 name인 오브젝트를 옮긴다 — capture_gazebo_dataset.py의 set_object_pose와
# 동일 패턴(그 스크립트는 즉석 월드용, 이건 world_name이 항상 warehouse_fuel로 고정).
def set_object_pose(node: Node, name: str, x: float, y: float, z: float, yaw: float) -> None:
    req = Pose()
    req.name = name
    req.position.x = x
    req.position.y = y
    req.position.z = z
    req.orientation.z = math.sin(yaw / 2)
    req.orientation.w = math.cos(yaw / 2)
    ok, _ = node.request(f"/world/{WORLD_NAME}/set_pose", req, Pose, Boolean, 2000)
    if not ok:
        raise RuntimeError(f"set_pose 실패: {name}")


def capture_frame(node: Node, camera_topic: str, retries: int = 3) -> np.ndarray:
    for attempt in range(retries):
        received = {}
        node.subscribe(Image, f"/{camera_topic}", lambda msg: received.setdefault("msg", msg))
        t0 = time.time()
        while "msg" not in received and time.time() - t0 < 5:
            time.sleep(0.02)
        node.unsubscribe(f"/{camera_topic}")
        if "msg" in received:
            msg = received["msg"]
            return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        print(f"  (재시도 {attempt + 1}/{retries}) {camera_topic}에서 프레임 수신 실패")
    raise TimeoutError(f"no frame from {camera_topic} after {retries} attempts")


# 걷는 액터의 현재 pose 계산에 필요한 sim_time을 /world/{world}/stats에서 읽는다 — dynamic_pose/info가
# 액터를 브로드캐스트하지 않는 걸 확인했으므로, stats의 sim_time + 우리가 직접 든 웨이포인트로 액터
# pose를 재현한다(actor_pose_at 참고).
def query_sim_time(node: Node, retries: int = 3) -> float:
    topic = f"/world/{WORLD_NAME}/stats"
    for attempt in range(retries):
        received = {}
        node.subscribe(WorldStatistics, topic, lambda msg: received.setdefault("msg", msg))
        t0 = time.time()
        while "msg" not in received and time.time() - t0 < 5:
            time.sleep(0.02)
        node.unsubscribe(topic)
        if "msg" in received:
            msg = received["msg"]
            return msg.sim_time.sec + msg.sim_time.nsec * 1e-9
        print(f"  (재시도 {attempt + 1}/{retries}) {topic}에서 sim_time 수신 실패")
    raise TimeoutError("sim_time 조회 실패")


# navi_factory.sdf를 읽기 전용으로 띄우고, 캡처 리그를 스폰한 뒤 트랙을 따라 순간이동하며
# 프레임+라벨을 저장한다. 라벨이 하나도 안 보이는 프레임도 버리지 않는다(빈 라벨 파일 = 실제
# 배경 hard negative). 인자로 정수를 주면(파일럿용) 캡처 지점 수를 그만큼으로 제한한다:
#   .venv-gz/bin/python src/capture_navi_factory_dataset.py 15
def main() -> None:
    pilot_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = f"{env.get('GZ_SIM_RESOURCE_PATH', '')}:{SIM_MODELS_DIR}"
    log_path = OUT_DIR / "navi_factory.log"
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(NAVI_FACTORY_SDF)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
    )
    try:
        time.sleep(6)  # navi_factory는 합성 월드보다 훨씬 무거워서 로드 시간을 넉넉히 준다
        node = Node()
        spawn_rig(node)
        time.sleep(2)  # 센서 플러그인 기동 대기

        route_waypoints = load_route_waypoints()
        path = resample_path(route_waypoints, STEP_DISTANCE)
        if pilot_limit is not None:
            path = path[:pilot_limit]
        num_passes = 1 if pilot_limit is not None else NUM_PASSES
        print(f"캡처 지점: {len(path)}개 x {num_passes}패스" + (" (파일럿)" if pilot_limit is not None else ""))

        # (heading에서의 yaw 오프셋, 라벨 계산에도 쓰이는 카메라 pose) 튜플 목록 — 매 지점마다
        # 전방 1장 + (SIDE_GLANCE_EVERY 번째마다) 좌우 글랜스 프레임을 만든다.
        def shots_at(idx: int, cam_pose: tuple) -> list[tuple]:
            cx, cy, cz, heading = cam_pose
            shots = [cam_pose]
            if idx % SIDE_GLANCE_EVERY == 0:
                shots += [(cx, cy, cz, heading + off) for off in SIDE_GLANCE_YAWS]
            return shots

        def compute_labels(cam_pose: tuple, sim_time: float) -> list[tuple]:
            labels = []
            for class_name, ox, oy, oz, oyaw, half in STATIC_OBJECTS:
                box = project_box(cam_pose, (ox, oy, oz), half, oyaw)
                if box is not None:
                    labels.append((SEMANTIC_CLASSES.index(class_name), *box))
            for actor_name in ACTOR_TRAJECTORIES:
                ax, ay, ayaw = actor_pose_at(actor_name, sim_time)
                box = project_box(cam_pose, (ax, ay, PERSON_HALF[2]), PERSON_HALF, ayaw)
                if box is not None:
                    labels.append((SEMANTIC_CLASSES.index("Person"), *box))
            return labels

        written = 0
        frame_no = 0
        total_shots = num_passes * sum(len(shots_at(i, (0, 0, 0, 0))) for i in range(len(path)))
        for pass_idx in range(num_passes):
            for idx, (x, y, heading) in enumerate(path):
                base_pose = camera_pose_at_track_point(x, y, heading)
                for cam_pose in shots_at(idx, base_pose):
                    set_object_pose(node, RIG_NAME, *cam_pose)
                    time.sleep(0.08)

                    sim_time = query_sim_time(node)
                    labels = compute_labels(cam_pose, sim_time)

                    try:
                        frame = capture_frame(node, RIG_CAMERA_TOPIC)
                    except TimeoutError as e:
                        print(f"  프레임 {frame_no:04d} 건너뜀: {e}")
                        continue

                    stem = f"navi_{frame_no:04d}"
                    PILImage.fromarray(frame).save(IMAGES_DIR / f"{stem}.png")
                    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in labels]
                    (LABELS_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
                    written += 1
                    frame_no += 1
                    if frame_no % 40 == 0:
                        print(f"  {frame_no}/{total_shots} 처리, {written}장 저장 (이번 프레임 라벨 {len(labels)}개)")

        print(f"navi_factory 캡처 완료: {written}장")
    finally:
        server.terminate()
        server.wait(timeout=15)


if __name__ == "__main__":
    main()
