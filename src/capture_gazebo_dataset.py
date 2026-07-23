# Last updated: 2026-07-23
'''
Gazebo 자체 렌더러로 YOLO 학습 데이터셋을 캡처한다.

이전 Blender 방식(render_gazebo_assets.py)을 대체한다: 실제 Gazebo 화면으로 검증해보니
Blender로 학습한 모델의 confidence가 0.98대에서 0.3 정도로 뚝 떨어지고 박스도 중복·헐렁하게
나와서, 로봇이 실제로 보게 될 조명/렌더러를 그대로 쓰기 위해 gz sim의 카메라 센서로 직접
학습 이미지를 뽑는다.

클래스마다 즉석에서 생성한 일회용 Gazebo 월드(바닥 + 태양광 + 고정 높이 카메라 3개 + 대상
오브젝트 하나)를 띄운다 — worlds/navi_factory/(Signal-Simulation의 실제 시뮬레이션 월드)는
이 스크립트가 절대 읽거나 쓰지 않는다. 카메라는 고정해두고 대신 대상 오브젝트 자체의 yaw
회전과 카메라 정면축 위 거리를 스윕한다(카메라가 고정된 물체 주위를 도는 것과 기하학적으로
동일). 각 프레임의 YOLO 라벨은 오브젝트의 알려진 pose/크기와 카메라의 알려진 intrinsics로
자동 계산한다 — 수동 라벨링 없음.

Gazebo 파이썬 바인딩이 필요한데 프로젝트 .venv 밖에 있다. 최초 1회 설정:
    python3.12 -m venv --system-site-packages .venv-gz
    .venv-gz/bin/pip install protobuf pillow numpy
이후 리포 루트에서 실행:
    .venv-gz/bin/python src/capture_gazebo_dataset.py
'''

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_MODELS_DIR = REPO_ROOT.parent / "Signal-Simulation" / "worlds" / "navi_factory" / "models"
OUT_DIR = REPO_ROOT / "data" / "raw" / "gazebo_capture"
WORLDS_DIR = OUT_DIR / "_worlds"  # 클래스별 일회용 월드 생성 위치 — navi_factory.sdf 아님

# 실제 3D 메시 자산(에셋 폴더명 기준) — 물리적으로 서로 다른 지오메트리라 캡처 시엔
# 하나씩 따로 스폰해야 하지만, 행동 판단(ADR-004)에선 "그 외 오브젝트 → 크기 기반 판단"으로
# 어차피 동일하게 처리되므로 여러 메시가 같은 의미 클래스로 묶일 수 있다 — 예를 들어
# ClutteringA/C/D_01은 전부 "잡동사니 더미"라는 같은 개념이라 별도 클래스로 쪼갤 이유가 없고,
# 오히려 하나로 합치면 클래스당 학습 데이터(형태 다양성)만 늘어난다.
#
# (의미 클래스, 카메라 거리 스케일링용 바운딩스피어 반지름, 라벨박스용 축별 half-extent,
# 수직 중심 높이) — 반지름/half-extent/높이는 전부 미터 단위, 이전 Blender 작업 때 각
# 메시의 bound_box로 측정해서 재사용(실시간 bbox 조회 대신). 반지름을 모든 축에 동일하게
# half-extent로 쓰면 라벨이 심하게 커진다(실제로 프레임 전체를 채우는 박스가 나오는 걸
# 확인함) — 딱 맞는 박스를 위해선 축별 실제 half-extent가 필요하다.
MESH_GEOMETRY = {
    # mesh_name:         (semantic_class, radius, half_x, half_y, half_z, center_z)
    "PalletJackB_01": ("PalletJack", 0.797, 0.576, 0.266, 0.482, 0.482),
    "Bucket_01": ("Bucket", 1.042, 0.471, 0.609, 0.702, 0.707),
    "ClutteringA_01": ("Cluttering", 1.554, 1.073, 0.991, 0.529, 0.573),
    "ClutteringC_01": ("Cluttering", 1.604, 0.885, 1.024, 0.862, 0.862),
    "ClutteringD_01": ("Cluttering", 1.164, 0.498, 0.751, 0.738, 0.737),
    "TrashCanC_01": ("TrashCan", 1.060, 0.723, 0.446, 0.634, 0.672),
}
TARGET_MESHES = list(MESH_GEOMETRY)

# 최종 YOLO 클래스(순서가 곧 class_id) — 여러 메시가 하나의 의미 클래스로 합쳐진 목록.
SEMANTIC_CLASSES = ["PalletJack", "Bucket", "Cluttering", "TrashCan"]

# 라벨박스 여유값: 위 half-extent는 Blender로 임포트(DAE->OBJ 변환)한 메시 기준이라 Gazebo
# 네이티브 렌더보다 살짝 작게 나온다(예: PalletJackB_01의 긴 손잡이가 잘림) — 완벽히
# 맞추려 하지 않고 패딩으로 보정한다.
BOX_PADDING = 1.2

IMG_SIZE = 640
HFOV = 1.05  # 라디안 — 아래 SDF의 카메라 <horizontal_fov>와 반드시 일치해야 함
CAMERAS = [("camera_low", 0.6), ("camera_mid", 1.0), ("camera_high", 1.5)]
DISTANCE_MULTIPLIERS = [1.5, 2.5, 4.0]
YAW_STEPS = 12


# 클래스 하나 + 고정 카메라 3대짜리 일회용 Gazebo 월드의 SDF 문자열을 만든다.
def world_sdf(class_name: str, world_name: str) -> str:
    mesh_uri = f"file://{SIM_MODELS_DIR}/aws_robomaker_warehouse_{class_name}"
    cameras_xml = "\n".join(
        f"""
    <model name="{name}">
      <pose>0 0 {height} 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <sensor name="cam" type="camera">
          <camera>
            <horizontal_fov>{HFOV}</horizontal_fov>
            <image><width>{IMG_SIZE}</width><height>{IMG_SIZE}</height></image>
            <clip><near>0.1</near><far>50</far></clip>
          </camera>
          <always_on>true</always_on>
          <update_rate>10</update_rate>
          <visualize>false</visualize>
          <topic>{name}</topic>
        </sensor>
      </link>
    </model>"""
        for name, height in CAMERAS
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="{world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>-0.3 0.3 -0.9</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    <model name="ground">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></visual>
      </link>
    </model>
    <include>
      <uri>{mesh_uri}</uri>
      <name>target</name>
      <pose>2 0 0 0 0 0</pose>
    </include>
{cameras_xml}
  </world>
</sdf>
"""


# set_pose 서비스를 호출해 대상 오브젝트를 (거리, 0, 높이) 위치로 옮기고 yaw만큼 회전시킨다.
def set_target_pose(node: Node, world_name: str, distance: float, center_z: float, yaw: float) -> None:
    req = Pose()
    req.name = "target"
    req.position.x = distance
    req.position.y = 0.0
    req.position.z = center_z
    req.orientation.z = math.sin(yaw / 2)
    req.orientation.w = math.cos(yaw / 2)
    ok, _ = node.request(f"/world/{world_name}/set_pose", req, Pose, Boolean, 2000)
    if not ok:
        raise RuntimeError("set_pose service call failed")


# 지정한 카메라 토픽에서 프레임 한 장을 구독해 받아오고, numpy 배열로 변환해 반환한다.
def capture_frame(node: Node, camera_topic: str) -> np.ndarray:
    received = {}
    sub = node.subscribe(Image, f"/{camera_topic}", lambda msg: received.setdefault("msg", msg))
    t0 = time.time()
    while "msg" not in received and time.time() - t0 < 5:
        time.sleep(0.02)
    node.unsubscribe(f"/{camera_topic}")
    if "msg" not in received:
        raise TimeoutError(f"no frame from {camera_topic}")
    msg = received["msg"]
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))


def yolo_label(
    distance: float, center_z: float, camera_z: float, yaw: float, half_x: float, half_y: float, half_z: float
) -> tuple | None:
    '''
    대상의 회전된 바운딩박스를 고정 카메라의 이미지 평면에 투영해 YOLO 라벨을 계산한다.

    카메라 몸체 좌표계: +X가 정면(대상 방향), +Y가 왼쪽, +Z가 위 — 생성된 카메라들의
    회전 없는(identity) 자세와 일치한다. 대상은 메시의 실제 축별 half-extent 크기의
    박스이며, 자체 yaw만큼 회전해서 (distance, 0, center_z)에 위치한다.
    '''
    focal = (IMG_SIZE / 2) / math.tan(HFOV / 2)
    cx = cy = IMG_SIZE / 2

    us, vs = [], []
    for sx in (-half_x, half_x):
        for sy in (-half_y, half_y):
            for sz in (-half_z, half_z):
                # 모서리 점을 yaw만큼 회전시킨 뒤 월드/카메라 좌표계로 배치
                wx = distance + sx * math.cos(yaw) - sy * math.sin(yaw)
                wy = 0.0 + sx * math.sin(yaw) + sy * math.cos(yaw)
                wz = center_z + sz - camera_z

                if wx <= 0.05:
                    continue  # 카메라 뒤쪽이거나 카메라 위치와 겹침
                us.append(cx - focal * (wy / wx))
                vs.append(cy - focal * (wz / wx))

    if not us:
        return None
    x_min, x_max = max(0.0, min(us)), min(IMG_SIZE, max(us))
    y_min, y_max = max(0.0, min(vs)), min(IMG_SIZE, max(vs))
    if x_max <= x_min or y_max <= y_min:
        return None

    box_cx, box_cy = (x_min + x_max) / 2 / IMG_SIZE, (y_min + y_max) / 2 / IMG_SIZE
    box_w, box_h = (x_max - x_min) / IMG_SIZE, (y_max - y_min) / IMG_SIZE
    return box_cx, box_cy, box_w, box_h


# 메시 하나에 대해 월드를 띄우고 카메라x거리x각도 조합을 전부 스윕하며 캡처+라벨링한다.
# 라벨에 쓰는 class_id는 이 메시가 속한 의미 클래스(SEMANTIC_CLASSES) 기준이다.
def capture_mesh(mesh_name: str, class_id: int) -> int:
    _, radius, half_x, half_y, half_z, center_z = MESH_GEOMETRY[mesh_name]
    world_name = f"capture_{mesh_name.lower()}"

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = WORLDS_DIR / f"{mesh_name}.sdf"
    sdf_path.write_text(world_sdf(mesh_name, world_name))

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = f"{env.get('GZ_SIM_RESOURCE_PATH', '')}:{SIM_MODELS_DIR}"

    log_path = OUT_DIR / "_worlds" / f"{mesh_name}.log"
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(sdf_path)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        time.sleep(3)  # 월드 로드 + 센서 플러그인 기동 대기
        node = Node()

        # 이미지는 메시별 폴더에 저장(디버깅 시 구분 쉽게), 라벨의 class_id만 의미 클래스로 합쳐짐
        images_dir = OUT_DIR / mesh_name / "images"
        labels_dir = OUT_DIR / mesh_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        idx = 0
        for camera_topic, camera_z in CAMERAS:
            for multiplier in DISTANCE_MULTIPLIERS:
                distance = multiplier * radius
                for step in range(YAW_STEPS):
                    yaw = 2 * math.pi * step / YAW_STEPS
                    set_target_pose(node, world_name, distance, center_z, yaw)
                    time.sleep(0.05)

                    label = yolo_label(
                        distance, center_z, camera_z, yaw,
                        half_x * BOX_PADDING, half_y * BOX_PADDING, half_z * BOX_PADDING,
                    )
                    if label is None:
                        idx += 1
                        continue

                    frame = capture_frame(node, camera_topic)
                    stem = f"{mesh_name}_{idx:04d}"
                    PILImage.fromarray(frame).save(images_dir / f"{stem}.png")
                    (labels_dir / f"{stem}.txt").write_text(
                        f"{class_id} {label[0]:.6f} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f}\n"
                    )
                    written += 1
                    idx += 1
        return written
    finally:
        server.terminate()
        server.wait(timeout=10)


# classes.txt(의미 클래스 목록)를 기록하고, 인자로 지정한 메시(없으면 전체)를 순서대로 캡처한다.
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("\n".join(SEMANTIC_CLASSES) + "\n")

    meshes = sys.argv[1:] or TARGET_MESHES
    for mesh_name in TARGET_MESHES:
        if mesh_name not in meshes:
            continue
        semantic_class = MESH_GEOMETRY[mesh_name][0]
        class_id = SEMANTIC_CLASSES.index(semantic_class)
        count = capture_mesh(mesh_name, class_id)
        print(f"{mesh_name} ({semantic_class}): {count} images")


if __name__ == "__main__":
    main()
