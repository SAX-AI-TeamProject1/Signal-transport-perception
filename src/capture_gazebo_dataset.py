# Last updated: 2026-08-03
'''
Gazebo 자체 렌더러로 YOLO 학습 데이터셋을 캡처한다.

이전 Blender 방식(render_gazebo_assets.py)을 대체한다: 실제 Gazebo 화면으로 검증해보니
Blender로 학습한 모델의 confidence가 0.98대에서 0.3 정도로 뚝 떨어지고 박스도 중복·헐렁하게
나와서, 로봇이 실제로 보게 될 조명/렌더러를 그대로 쓰기 위해 gz sim의 카메라 센서로 직접
학습 이미지를 뽑는다.

라벨(바운딩박스)은 더 이상 메시 half-extent를 손으로 측정해 투영 계산하지 않는다. gz-sim이
자체 제공하는 `boundingbox_camera` 센서(+ `gz-sim-label-system` 플러그인)를 각 RGB 카메라와
동일한 pose로 같이 띄워서, 렌더러가 직접 계산한 픽셀 단위 박스를 그대로 읽어 쓴다 — 이 프로젝트는
Blender를 쓰지 않으므로, 새 클래스를 추가할 때도 메시 치수를 별도로 측정할 필요가 없다.

카메라 거리/높이(예전엔 메시별 half-extent로 미리 계산해둔 "반지름"·"중심 높이" 상수)도 더 이상
사람이 채워 넣지 않는다. `calibrate_target()`이 캡처 시작 직전에 boundingbox_camera 피드백만
보면서 (1) 물체가 프레임에 들어올 때까지 거리를 좁히고 (2) 세로 중앙에 오도록 높이를 보정하고
(3) 목표 프레임 비율이 되도록 거리를 다시 맞추는 짧은 루프를 돌려 실측한다. 그래서 새 클래스를
등록할 때 필요한 정보는 "어떤 메시를 스폰할지"뿐이다.

클래스마다 즉석에서 생성한 일회용 Gazebo 월드(바닥 + 태양광 + 고정 높이 카메라 3개 + 대상
오브젝트 하나)를 띄운다 — worlds/navi_factory/(Signal-Simulation의 실제 시뮬레이션 월드)는
이 스크립트가 절대 읽거나 쓰지 않는다. 카메라는 고정해두고 대신 대상 오브젝트 자체의 yaw
회전과 카메라 정면축 위 거리를 스윕한다(카메라가 고정된 물체 주위를 도는 것과 기하학적으로
동일).

Gazebo 파이썬 바인딩이 필요한데 프로젝트 .venv 밖에 있다. 최초 1회 설정:
    python3.12 -m venv --system-site-packages .venv-gz
    .venv-gz/bin/pip install protobuf pillow numpy
이후 리포 루트에서 실행:
    .venv-gz/bin/python src/capture_gazebo_dataset.py
'''

import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_MODELS_DIR = REPO_ROOT.parent / "Signal-Simulation" / "worlds" / "navi_factory" / "models"
OUT_DIR = REPO_ROOT / "data" / "raw" / "gazebo_capture"
WORLDS_DIR = OUT_DIR / "_worlds"  # 클래스별 일회용 월드 생성 위치 — navi_factory.sdf 아님

# 자체 model.sdf가 있어 <include>로 통째로 스폰 가능한 에셋(키 = Signal-Simulation의
# worlds/navi_factory/models/ 아래 실제 폴더명) → semantic_class. 카메라 거리/높이는 더 이상
# 여기 적어두지 않는다 — calibrate_target()이 실시간으로 알아낸다.
PACKAGE_MESHES = {
    "aws_robomaker_warehouse_PalletJackB_01": "PalletJack",
    "aws_robomaker_warehouse_Bucket_01": "Bucket",
    "aws_robomaker_warehouse_ClutteringA_01": "Cluttering",
    "aws_robomaker_warehouse_ClutteringC_01": "Cluttering",
    "aws_robomaker_warehouse_ClutteringD_01": "Cluttering",
    "aws_robomaker_warehouse_TrashCanC_01": "TrashCan",
}

# 자체 model.sdf가 없는 순수 메시 에셋(navi_factory.sdf가 이 오브젝트들을 실제로 배치할 때 쓰는
# <visual> 구성을 그대로 재사용한다 — mesh 경로/visual pose 모두 Signal-Simulation의
# navi_factory.sdf에서 그대로 가져온 값). navi_factory.sdf에서 이 오브젝트들은 전부
# <static>true</static> 배경 소품으로 배치돼 있어 물리 충돌이 필요 없으므로 별도 collision은
# 만들지 않는다. (semantic_class, mesh 상대경로, visual pose) — 카메라 거리/높이는 여기 없음.
RAW_MESH_MESHES = {
    # mesh_name:      (semantic_class, mesh 상대경로,                        visual pose)
    "StandingPerson": ("Person", "StandingPerson/meshes/standing.dae", "0 0 0.02 0.04 0 0"),
    "WalkingPerson": ("Person", "WalkingPerson/meshes/walking.dae", "0 0 0.02 0.04 0 0"),
    "CasualFemale": ("Person", "CasualFemale/meshes/casual_female.dae", "0 0 0.02 0.04 0 0"),
    "Forklift": ("Vehicle", "Forklift/base_visual.glb", "0 0 0 0 0 0"),
    # navi_factory.sdf가 실제로 <actor> 애니메이션(걷는 방문객 7명)에 쓰는 메시 두 종 — 애니메이션
    # 자체는 재현하지 않고 이 메시를 정지 포즈로 스폰만 한다(단일 프레임 캡처라 걷는 동작 자체는
    # 필요 없음). visual pose는 다른 사람 메시들과 동일하게 지면 보정만 준다.
    "MaleVisitor": ("Person", "MaleVisitor/meshes/MaleVisitorWalk.dae", "0 0 0.02 0.04 0 0"),
    "FemaleVisitorWalk": ("Person", "FemaleVisitorWalk/meshes/FemaleVisitorWalk.dae", "0 0 0.02 0.04 0 0"),
}

TARGET_MESHES = list(PACKAGE_MESHES) + list(RAW_MESH_MESHES)

# 최종 YOLO 클래스(순서가 곧 class_id) — 여러 메시가 하나의 의미 클래스로 합쳐진 목록.
# 기존 4개 뒤에 새 클래스를 추가해 기존 class_id(0~3)는 그대로 유지한다.
SEMANTIC_CLASSES = ["PalletJack", "Bucket", "Cluttering", "TrashCan", "Person", "Vehicle"]

IMG_SIZE = 640
HFOV = 1.05  # 라디안 — 아래 SDF의 카메라 <horizontal_fov>와 반드시 일치해야 함
# (name, height, pitch) — pitch는 라디안, 양수면 아래로 기울임(navi_factory.sdf GUI 기본 조감뷰의
# <camera_pose>...1.006...> 부호와 동일 convention). 단일 오브젝트 캘리브레이션/스윕(capture_mesh)은
# 이 3대(지상 레벨 촬영)만 쓴다 — 캘리브레이션이 camera_mid 기준 눈높이를 가정하고 있어서, 아래로
# 꺾인 카메라를 섞으면 물체가 프레임 밖으로 벗어나거나 캘리브레이션이 어긋날 수 있다.
CAMERAS = [("camera_low", 0.6, 0.0), ("camera_mid", 1.0, 0.0), ("camera_high", 1.5, 0.0)]
# navi_factory.sdf GUI 기본 조감뷰(0 48 24, pitch 1.006)처럼 로봇 카메라가 아닌 "관제/모니터링"
# 각도까지 학습에 포함시키기 위한 광각 카메라 — 멀티오브젝트 씬(capture_multi_scene)에서만 쓴다.
# 높이/피치는 시작값이고, 파일럿 캡처 이미지를 test_picture와 비교하며 튜닝할 것.
OVERVIEW_CAMERA = ("camera_overview", 10.0, 0.6)
MULTI_SCENE_CAMERAS = CAMERAS + [OVERVIEW_CAMERA]  # 멀티오브젝트 씬은 지상 3대 + 조감 1대
ALL_CAMERAS = MULTI_SCENE_CAMERAS  # 월드 SDF에는 항상 다 심어둔다 — capture_mesh는 그중 3개만 구독
CAMERA_HEIGHTS = {name: height for name, height, _ in ALL_CAMERAS}
CALIBRATION_CAMERA = "camera_mid"  # 거리/높이 캘리브레이션 기준 카메라
TARGET_FILL = 0.5  # 캘리브레이션 목표: 박스 세로 길이가 프레임의 이 비율을 채우도록
DISTANCE_RATIOS = [0.5, 0.7, 1.0, 1.6, 3.0, 6.0]  # 캘리브레이션된 기준 거리에 대한 근~원 배율
# 실제 배포 화면(창고 전체를 내려다보는 먼 시점)에서 물체가 프레임의 5~10%밖에 안 되는데,
# 기존 [0.7, 1.0, 1.6]은 전부 프레임의 30~50% 근처에 몰려있어 "작은 스케일" 자체를 학습에서
# 아예 본 적이 없었다(같은 물체를 8배 멀리서 캡처하니 탐지 0개로 확인). 모든 거리를 다 외우게
# 하려는 게 아니라, 이 범위 안에서 크기 다양성을 보여줘 스케일 불변성이 일반화되게 하는 것.
YAW_STEPS = 12
MIN_BOX_PIXELS = 4  # 이보다 작은 박스(거의 안 보이는 프레임)는 노이즈로 버린다

# 라벨 없는 배경 클러터용 정적 소품(자체 model.sdf가 있는 것만 — target_sdf의 PACKAGE_MESHES
# 스폰 방식을 그대로 재사용). 실제 창고 화면(선반/기둥/박스 무더기)처럼 복잡한 배경에서도
# 오탐지 없이 대상 클래스만 잡아내도록, 단일 오브젝트 캡처에는 없던 시각적 잡음을 넣어준다.
CLUTTER_MESHES = [
    "aws_robomaker_warehouse_ShelfD_01",
    "aws_robomaker_warehouse_ShelfE_01",
    "aws_robomaker_warehouse_WallB_01",
    "shelf_big",  # navi_factory.sdf가 실제로 쓰는 대형 선반(shelf_big_east_1 등) — model.sdf 있음
]

# model.sdf 없는 raw mesh 클러터(하드 네거티브) — navi_factory.sdf에서 실제로 오탐지를 유발한
# ControlConsole("포장기계"로 보이던 소품)/ElectricalPanelBareModel을 그대로 가져온다. 학습이
# 한 번도 못 본 물체를 가장 비슷한 학습 클래스로 욱여넣어 분류하던 게 오탐지 원인이었으므로,
# 이 소품들을 라벨 없이 노출시켜 "이건 어떤 클래스도 아니다"를 배우게 한다. (mesh 상대경로,
# visual pose, mesh scale) — scale은 navi_factory.sdf의 실제 배치값(1.5배)을 그대로 가져옴.
HARD_NEGATIVE_MESHES = {
    "ControlConsole": ("ControlConsole/meshes/console.dae", "0 0 0 0 0 0", 1.5),
    "ElectricalPanelBareModel": (
        "ElectricalPanelBareModel/meshes/Underwater-Frame.dae", "0 0 0 0 0 1.5708", 1.5,
    ),
}
SCENE_OBJECT_COUNT = (2, 4)  # 씬 하나에 스폰할 라벨 대상 오브젝트 개수 범위(양끝 포함)
SCENE_CLUTTER_COUNT = (2, 3)  # 씬 하나에 스폰할 무라벨 클러터 개수 범위(양끝 포함)
SCENE_FOV_MARGIN = 0.8  # 오브젝트를 HFOV의 이 비율 안에 배치(가장자리 여유)
NUM_SCENES = 250  # 멀티오브젝트 씬 개수(MULTI_SCENE_CAMERAS 4대라 총 이미지는 최대 NUM_SCENES*4).
# ADR-013의 150에서 늘림 — 창고 배경/하드 네거티브 클러터/조감 카메라가 새로 추가되면서 씬 조합
# 다양성 자체가 커져서, 같은 150개로는 새 조합들을 고루 못 볼 수 있어서다.


def semantic_class_of(mesh_name: str) -> str:
    if mesh_name in PACKAGE_MESHES:
        return PACKAGE_MESHES[mesh_name]
    return RAW_MESH_MESHES[mesh_name][0]


# 대상 오브젝트 하나를 스폰하는 SDF 조각을 만든다. class_id는 gz-sim-label-system 플러그인에
# 그대로 박아서, boundingbox_camera가 이 오브젝트를 그 라벨로 리포트하게 한다. name을 바꿔주면
# 한 월드에 여러 개를 동시에 스폰할 수 있다(멀티오브젝트 씬용).
def target_sdf(mesh_name: str, class_id: int, name: str = "target", pose: str = "2 0 0 0 0 0") -> str:
    label_plugin = f"""
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
        <label>{class_id}</label>
      </plugin>"""

    if mesh_name in PACKAGE_MESHES:
        mesh_uri = f"file://{SIM_MODELS_DIR}/{mesh_name}"
        return f"""
    <include>
      <uri>{mesh_uri}</uri>
      <name>{name}</name>
      <pose>{pose}</pose>{label_plugin}
    </include>"""

    _, mesh_relpath, visual_pose = RAW_MESH_MESHES[mesh_name]
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <visual name="visual">
          <pose>{visual_pose}</pose>
          <geometry>
            <mesh><uri>model://{mesh_relpath}</uri></mesh>
          </geometry>
        </visual>
      </link>{label_plugin}
    </model>"""


# CLUTTER_MESHES/HARD_NEGATIVE_MESHES 중 하나를 라벨 없이 스폰하는 SDF 조각을 만든다 —
# boundingbox_camera가 Label 플러그인이 없는 오브젝트는 리포트하지 않으므로, 이 소품들은 배경
# 잡음으로만 작용한다. target_sdf()와 동일하게 package(model.sdf 있음)/raw mesh 두 갈래를 쓴다.
def clutter_sdf(mesh_name: str, name: str, pose: str) -> str:
    if mesh_name in HARD_NEGATIVE_MESHES:
        mesh_relpath, visual_pose, scale = HARD_NEGATIVE_MESHES[mesh_name]
        return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <visual name="visual">
          <pose>{visual_pose}</pose>
          <geometry>
            <mesh>
              <scale>{scale} {scale} {scale}</scale>
              <uri>model://{mesh_relpath}</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>"""

    mesh_uri = f"file://{SIM_MODELS_DIR}/{mesh_name}"
    return f"""
    <include>
      <uri>{mesh_uri}</uri>
      <name>{name}</name>
      <pose>{pose}</pose>
    </include>"""


# 고정 카메라 ALL_CAMERAS(RGB + boundingbox_camera 쌍) SDF 조각 — 단일 오브젝트 월드와
# 멀티오브젝트 씬 월드가 동일하게 재사용한다. 물리적으로는 모든 월드에 4대(지상 3대 + 조감 1대)가
# 다 심어지지만, capture_mesh()는 그중 지상 3대(CAMERAS)만 구독한다 — 안 쓰는 조감 카메라가 있어도
# 렌더 비용만 조금 늘 뿐 해는 없다.
def cameras_sdf() -> str:
    return "\n".join(
        f"""
    <model name="{name}">
      <pose>0 0 {height} 0 {pitch} 0</pose>
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
        <sensor name="bbox" type="boundingbox_camera">
          <camera>
            <box_type>visible_2d</box_type>
            <horizontal_fov>{HFOV}</horizontal_fov>
            <image><width>{IMG_SIZE}</width><height>{IMG_SIZE}</height></image>
            <clip><near>0.1</near><far>50</far></clip>
          </camera>
          <always_on>true</always_on>
          <update_rate>10</update_rate>
          <visualize>false</visualize>
          <topic>{name}_bbox</topic>
        </sensor>
      </link>
    </model>"""
        for name, height, pitch in ALL_CAMERAS
    )


# navi_factory.sdf의 창고 건물 메시(기둥/벽/바닥이 통째로 구워진 단일 메시)를 라벨 없는 정적
# 배경으로 그대로 복제해 넣는다 — navi_factory.sdf 자체는 여전히 읽거나 쓰지 않는다(같은
# SIM_MODELS_DIR 애셋을 우리 즉석 월드에서도 참조하는 것뿐). scale/pose는 navi_factory.sdf의
# 실제 배치값과 동일하게 맞췄다 — 기둥을 낱개 애셋으로 뽑아 쓸 수 없어(건물 메시에 구워짐),
# 배경으로 통째로 포함하는 게 유일한 방법이다.
WAREHOUSE_BACKDROP_SDF = """
    <model name="warehouse_backdrop">
      <static>true</static>
      <pose>0 0 -0.1002 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <mesh>
              <scale>2 2 2</scale>
              <uri>model://warehouse_mohamedazimal27_2x/meshes/warehouse.dae</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>"""


# 월드 공통 껍데기(physics/plugin/조명/바닥) + 카메라 + 본문(단일 오브젝트든 멀티오브젝트 씬이든)을
# 합쳐 완성한 SDF 문자열을 만든다. include_ground=False면 단순 평면 바닥 대신 호출자가 body_xml에
# 이미 넣어둔 실제 배경(WAREHOUSE_BACKDROP_SDF 등)을 쓴다는 뜻 — 평면과 창고 메시 바닥이 겹쳐
# z-fighting 나는 것을 피하기 위함.
def base_world_sdf(world_name: str, body_xml: str, include_ground: bool = True) -> str:
    ground_xml = """
    <model name="ground">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></visual>
      </link>
    </model>""" if include_ground else ""
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
    </light>{ground_xml}
{body_xml}
{cameras_sdf()}
  </world>
</sdf>
"""


# 클래스 하나 + 고정 카메라(ALL_CAMERAS, capture_mesh는 그중 지상 3대만 씀)짜리 일회용 Gazebo
# 월드의 SDF 문자열을 만든다. 카메라마다 RGB 이미지용 camera 센서와, 자동 라벨용
# boundingbox_camera 센서를 같은 pose로 짝지어 넣는다.
def world_sdf(mesh_name: str, world_name: str, class_id: int) -> str:
    return base_world_sdf(world_name, target_sdf(mesh_name, class_id))


# set_pose 서비스를 호출해 이름이 name인 오브젝트를 (x, y, z) 위치로 옮기고 yaw만큼 회전시킨다.
def set_object_pose(node: Node, world_name: str, name: str, x: float, y: float, z: float, yaw: float) -> None:
    req = Pose()
    req.name = name
    req.position.x = x
    req.position.y = y
    req.position.z = z
    req.orientation.z = math.sin(yaw / 2)
    req.orientation.w = math.cos(yaw / 2)
    ok, _ = node.request(f"/world/{world_name}/set_pose", req, Pose, Boolean, 2000)
    if not ok:
        raise RuntimeError(f"set_pose service call failed for {name}")


# 단일 오브젝트 월드에서 "target"이라는 이름의 오브젝트를 (거리, 0, 높이)로 옮기는
# set_object_pose의 단축형 — calibrate_target/capture_mesh가 쓴다.
def set_target_pose(node: Node, world_name: str, distance: float, center_z: float, yaw: float) -> None:
    set_object_pose(node, world_name, "target", distance, 0.0, center_z, yaw)


# 지정한 카메라 토픽에서 프레임 한 장을 구독해 받아오고, numpy 배열로 변환해 반환한다.
# gz-transport 구독이 간헐적으로 한 번 씹히는 경우가 있어(수백 장 캡처하다 보면 실제로 발생),
# 몇 번 재시도한 뒤에도 안 오면 그때 포기한다.
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


# 지정한 boundingbox_camera 토픽에서 메시지 하나를 구독해 받아온다(없으면 None) — capture_bbox와
# capture_bbox_all이 공유하는 저수준 로직.
def _read_bbox_msg(node: Node, bbox_topic: str) -> AnnotatedAxisAligned2DBox_V | None:
    received = {}
    node.subscribe(
        AnnotatedAxisAligned2DBox_V, f"/{bbox_topic}", lambda msg: received.setdefault("msg", msg)
    )
    t0 = time.time()
    while "msg" not in received and time.time() - t0 < 5:
        time.sleep(0.02)
    node.unsubscribe(f"/{bbox_topic}")
    return received.get("msg")


def _normalize_box(annotated) -> tuple | None:
    x_min = max(0.0, annotated.box.min_corner.x)
    y_min = max(0.0, annotated.box.min_corner.y)
    x_max = min(float(IMG_SIZE), annotated.box.max_corner.x)
    y_max = min(float(IMG_SIZE), annotated.box.max_corner.y)
    if x_max - x_min < MIN_BOX_PIXELS or y_max - y_min < MIN_BOX_PIXELS:
        return None
    box_cx, box_cy = (x_min + x_max) / 2 / IMG_SIZE, (y_min + y_max) / 2 / IMG_SIZE
    box_w, box_h = (x_max - x_min) / IMG_SIZE, (y_max - y_min) / IMG_SIZE
    return box_cx, box_cy, box_w, box_h


# 지정한 boundingbox_camera 토픽에서 class_id 라벨이 붙은 박스를 찾아 YOLO 정규화 좌표로 반환한다.
# gz-sim 렌더러가 직접 계산한 픽셀 박스이므로, 이 프로젝트는 더 이상 메시 half-extent를 투영
# 계산하지 않는다. 단일 오브젝트 월드용(라벨이 하나뿐이라고 가정) — 여러 오브젝트가 동시에 있는
# 씬에는 capture_bbox_all을 쓴다.
def capture_bbox(node: Node, bbox_topic: str, class_id: int) -> tuple | None:
    msg = _read_bbox_msg(node, bbox_topic)
    if msg is None:
        return None
    for annotated in msg.annotated_box:
        if annotated.label != class_id:
            continue
        box = _normalize_box(annotated)
        if box is not None:
            return box
    return None


# 씬에 라벨 붙은 오브젝트가 여러 개 있을 때, 보이는 것 전부를 (class_id, cx, cy, w, h) 리스트로
# 반환한다(멀티오브젝트 씬용). 라벨 없는 클러터 소품은 애초에 Label 플러그인이 없어 이 메시지에
# 안 잡힌다.
def capture_bbox_all(node: Node, bbox_topic: str) -> list[tuple]:
    msg = _read_bbox_msg(node, bbox_topic)
    if msg is None:
        return []
    results = []
    for annotated in msg.annotated_box:
        box = _normalize_box(annotated)
        if box is not None:
            results.append((annotated.label, *box))
    return results


# 메시 크기를 사람이 미리 재두지 않고, boundingbox_camera 피드백만 보면서 실시간으로
# (지면 위 중심 높이, 기준 거리)를 찾는다.
#   1) 물체가 기준 카메라 프레임에 들어올 때까지 거리를 좁혀 들어간다.
#   2) 박스가 세로 중앙에 오도록 높이를 보정한다(원근 비례로 근사 조정, 몇 번 반복해 수렴).
#   3) 박스 세로 길이가 TARGET_FILL 비율이 되도록 거리를 다시 맞춘다.
def calibrate_target(node: Node, world_name: str, class_id: int) -> tuple[float, float]:
    camera_z = CAMERA_HEIGHTS[CALIBRATION_CAMERA]
    bbox_topic = f"{CALIBRATION_CAMERA}_bbox"
    z = camera_z  # 초기 추정치: 기준 카메라 높이 근처에서 시작
    distance = 8.0

    box = None
    for _ in range(12):
        set_target_pose(node, world_name, distance, z, 0.0)
        time.sleep(0.05)
        box = capture_bbox(node, bbox_topic, class_id)
        if box is not None and box[3] >= 0.08:
            break
        distance *= 0.75
        if distance < 0.2:
            break
    if box is None:
        raise RuntimeError(f"calibration failed: target(class {class_id}) never entered frame")

    for _ in range(6):
        set_target_pose(node, world_name, distance, z, 0.0)
        time.sleep(0.05)
        box = capture_bbox(node, bbox_topic, class_id)
        if box is None:
            break
        error = box[1] - 0.5  # box_cy가 0.5보다 크면(프레임 아래쪽) 물체를 더 높여야 한다
        if abs(error) < 0.04:
            break
        z += error * distance

    for _ in range(6):
        set_target_pose(node, world_name, distance, z, 0.0)
        time.sleep(0.05)
        box = capture_bbox(node, bbox_topic, class_id)
        if box is None or box[3] <= 0:
            break
        distance *= box[3] / TARGET_FILL
        if abs(box[3] - TARGET_FILL) < 0.05:
            break

    return z, distance


# 메시 하나에 대해 월드를 띄우고 카메라x거리x각도 조합을 전부 스윕하며 캡처+라벨링한다.
# 라벨에 쓰는 class_id는 이 메시가 속한 의미 클래스(SEMANTIC_CLASSES) 기준이다.
def capture_mesh(mesh_name: str, class_id: int) -> int:
    world_name = f"capture_{mesh_name.lower()}"

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = WORLDS_DIR / f"{mesh_name}.sdf"
    sdf_path.write_text(world_sdf(mesh_name, world_name, class_id))

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
        center_z, base_distance = calibrate_target(node, world_name, class_id)

        # 이미지는 메시별 폴더에 저장(디버깅 시 구분 쉽게), 라벨의 class_id만 의미 클래스로 합쳐짐
        images_dir = OUT_DIR / mesh_name / "images"
        labels_dir = OUT_DIR / mesh_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        idx = 0
        for camera_topic, _, _ in CAMERAS:
            for ratio in DISTANCE_RATIOS:
                distance = ratio * base_distance
                for step in range(YAW_STEPS):
                    yaw = 2 * math.pi * step / YAW_STEPS
                    set_target_pose(node, world_name, distance, center_z, yaw)
                    time.sleep(0.05)

                    label = capture_bbox(node, f"{camera_topic}_bbox", class_id)
                    if label is None:
                        idx += 1
                        continue

                    try:
                        frame = capture_frame(node, camera_topic)
                    except TimeoutError as e:
                        # 센서 구독이 간헐적으로 씹히는 정도로는 프레임 하나만 건너뛰고
                        # 나머지 스윕(이 메시의 남은 각도/거리, 그리고 다음 메시들)은 계속 진행한다.
                        print(f"  프레임 {idx:04d} 건너뜀: {e}")
                        idx += 1
                        continue
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


# 멀티오브젝트 씬에 쓸 (center_z, base_distance)를 얻으려고, 이 메시 하나만 있는 임시 단일
# 오브젝트 월드를 잠깐 띄워 calibrate_target()을 재사용한다.
def calibrate_mesh(mesh_name: str) -> tuple[float, float]:
    class_id = SEMANTIC_CLASSES.index(semantic_class_of(mesh_name))
    world_name = f"calib_{mesh_name.lower()}"

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = WORLDS_DIR / f"calib_{mesh_name}.sdf"
    sdf_path.write_text(world_sdf(mesh_name, world_name, class_id))

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = f"{env.get('GZ_SIM_RESOURCE_PATH', '')}:{SIM_MODELS_DIR}"
    log_path = OUT_DIR / "_worlds" / f"calib_{mesh_name}.log"
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(sdf_path)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
    )
    try:
        time.sleep(3)
        node = Node()
        return calibrate_target(node, world_name, class_id)
    finally:
        server.terminate()
        server.wait(timeout=10)


# 씬 하나에 스폰할 대상(라벨 있는) 오브젝트들을 무작위로 고르고 배치한다. 카메라 시야 안에서
# 서로 다른 각도 슬롯에 하나씩 배정해 화면에 고르게 퍼지도록 하고(다 겹쳐서 한 덩어리로 안 보이게),
# 거리는 DISTANCE_RATIOS 중 무작위로 골라 스케일 다양성도 그대로 유지한다.
def random_scene_objects(calibration: dict[str, tuple[float, float]]) -> list[tuple]:
    mesh_pool = list(calibration.keys())
    n = random.randint(*SCENE_OBJECT_COUNT)
    layout = []
    for i in range(n):
        mesh_name = random.choice(mesh_pool)
        class_id = SEMANTIC_CLASSES.index(semantic_class_of(mesh_name))
        center_z, base_distance = calibration[mesh_name]

        slot_width = (HFOV * SCENE_FOV_MARGIN) / n
        angle = -HFOV * SCENE_FOV_MARGIN / 2 + slot_width * (i + 0.5)
        angle += random.uniform(-slot_width * 0.3, slot_width * 0.3)
        distance = base_distance * random.choice(DISTANCE_RATIOS)

        x, y = distance * math.cos(angle), distance * math.sin(angle)
        yaw = random.uniform(0, 2 * math.pi)
        layout.append((mesh_name, class_id, f"obj{i}", x, y, center_z, yaw))
    return layout


# 라벨 없는 배경 소품을 무작위로 흩뿌린다 — 대상 오브젝트보다 먼 거리에 배치해서 가리지 않게 한다.
# CLUTTER_MESHES(선반류) + HARD_NEGATIVE_MESHES(포장기계류)를 한 풀로 섞어서, 매 씬마다 둘 다
# 나올 수 있게 한다.
CLUTTER_POOL = CLUTTER_MESHES + list(HARD_NEGATIVE_MESHES)


def random_scene_clutter() -> list[tuple]:
    n = random.randint(*SCENE_CLUTTER_COUNT)
    layout = []
    for i in range(n):
        mesh_name = random.choice(CLUTTER_POOL)
        distance = random.uniform(6.0, 12.0)
        angle = random.uniform(-HFOV, HFOV)
        yaw = random.uniform(0, 2 * math.pi)
        x, y = distance * math.cos(angle), distance * math.sin(angle)
        layout.append((mesh_name, f"clutter{i}", x, y, yaw))
    return layout


def multi_scene_sdf(objects: list[tuple], clutter: list[tuple], world_name: str) -> str:
    objects_xml = "\n".join(
        target_sdf(mesh_name, class_id, name, f"{x} {y} {z} 0 0 {yaw}")
        for mesh_name, class_id, name, x, y, z, yaw in objects
    )
    clutter_xml = "\n".join(
        clutter_sdf(mesh_name, name, f"{x} {y} 0 0 0 {yaw}") for mesh_name, name, x, y, yaw in clutter
    )
    body = WAREHOUSE_BACKDROP_SDF + "\n" + objects_xml + "\n" + clutter_xml
    return base_world_sdf(world_name, body, include_ground=False)


# 오브젝트 여러 개 + 배경 클러터를 무작위 배치한 씬 하나를 만들어 MULTI_SCENE_CAMERAS(지상 3대 +
# 조감 1대)로 캡처한다. 단일
# 오브젝트 캡처와 달리 이미지 한 장에 라벨이 여러 줄(오브젝트마다 한 줄) 있을 수 있다 —
# boundingbox_camera가 리포트하는, 라벨 붙은 오브젝트 전부를 그대로 받아 적는다.
def capture_multi_scene(scene_idx: int, calibration: dict[str, tuple[float, float]]) -> int:
    objects = random_scene_objects(calibration)
    clutter = random_scene_clutter()
    world_name = f"scene_{scene_idx}"

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = WORLDS_DIR / f"scene_{scene_idx:04d}.sdf"
    sdf_path.write_text(multi_scene_sdf(objects, clutter, world_name))

    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = f"{env.get('GZ_SIM_RESOURCE_PATH', '')}:{SIM_MODELS_DIR}"
    log_path = OUT_DIR / "_worlds" / f"scene_{scene_idx:04d}.log"
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(sdf_path)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
    )
    try:
        time.sleep(3)  # 월드 로드 + 센서 플러그인 기동 대기(오브젝트는 스폰 시점 pose로 이미 배치됨)
        node = Node()

        images_dir = OUT_DIR / "_multi_scene" / "images"
        labels_dir = OUT_DIR / "_multi_scene" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for camera_topic, _, _ in MULTI_SCENE_CAMERAS:
            boxes = capture_bbox_all(node, f"{camera_topic}_bbox")
            if not boxes:
                continue
            try:
                frame = capture_frame(node, camera_topic)
            except TimeoutError as e:
                print(f"  씬 {scene_idx:04d}/{camera_topic} 건너뜀: {e}")
                continue
            stem = f"scene_{scene_idx:04d}_{camera_topic}"
            PILImage.fromarray(frame).save(images_dir / f"{stem}.png")
            lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in boxes]
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            written += 1
        return written
    finally:
        server.terminate()
        server.wait(timeout=10)


# 대상 오브젝트+배경 클러터를 섞은 멀티오브젝트 씬을 num_scenes개 캡처한다(MULTI_SCENE_CAMERAS
# 4대라 이미지는 최대 num_scenes*4장). 메시별 (center_z, base_distance)는 한 번씩만 캘리브레이션해서 재사용—
# 씬마다 다시 재는 게 아니라, 단일 오브젝트 캡처 때 이미 검증된 값을 그대로 가져다 쓰는 것.
def capture_multi_scenes(num_scenes: int = NUM_SCENES) -> int:
    print("멀티오브젝트 씬 캘리브레이션 중...")
    calibration = {}
    for mesh_name in TARGET_MESHES:
        try:
            calibration[mesh_name] = calibrate_mesh(mesh_name)
        except Exception as e:
            print(f"  {mesh_name} 캘리브레이션 실패, 씬 배치에서 제외: {e}")
    if not calibration:
        raise RuntimeError("모든 메시 캘리브레이션 실패 — 멀티오브젝트 씬을 만들 수 없음")

    total = 0
    for scene_idx in range(num_scenes):
        try:
            total += capture_multi_scene(scene_idx, calibration)
        except Exception as e:
            print(f"  씬 {scene_idx:04d} 실패: {e}")
    return total


# classes.txt(의미 클래스 목록)를 기록하고, 인자로 지정한 메시(없으면 전체)를 순서대로 캡처한다.
# 인자를 하나도 안 주면(전체 실행) 단일 오브젝트 캡처가 끝난 뒤 멀티오브젝트 씬 캡처도 이어서
# 한다 — 특정 메시 이름만 넘기면(부분 재실행) 멀티오브젝트 씬은 건너뛴다. "--multi"만 넘기면
# 멀티오브젝트 씬만 캡처한다(단일 오브젝트 데이터는 이미 있는 상태에서 이것만 다시 만들 때 씀).
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("\n".join(SEMANTIC_CLASSES) + "\n")

    args = sys.argv[1:]
    if args == ["--multi"]:
        count = capture_multi_scenes()
        print(f"멀티오브젝트 씬: {count} images")
        return

    meshes = args or TARGET_MESHES
    for mesh_name in TARGET_MESHES:
        if mesh_name not in meshes:
            continue
        semantic_class = semantic_class_of(mesh_name)
        class_id = SEMANTIC_CLASSES.index(semantic_class)
        try:
            count = capture_mesh(mesh_name, class_id)
            print(f"{mesh_name} ({semantic_class}): {count} images")
        except Exception as e:
            # 메시 하나가 통째로 실패해도(예: 캘리브레이션 실패) 나머지 메시는 계속 캡처한다 —
            # 이미 끝난 메시들 데이터를 날리고 처음부터 다시 돌릴 필요가 없게.
            print(f"{mesh_name} ({semantic_class}): 실패 — {e}")

    if not args:
        try:
            count = capture_multi_scenes()
            print(f"멀티오브젝트 씬: {count} images")
        except Exception as e:
            print(f"멀티오브젝트 씬 캡처 실패: {e}")


if __name__ == "__main__":
    main()
