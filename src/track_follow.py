# Last updated: 2026-07-31
'''
카메라 프레임에서 바닥에 그려진 트랙(navi_factory의 주황색 대시라인)을 인식해, 로봇 바로 앞
트랙의 좌우 오프셋과 곡률을 추정한다.

분기점/반환점/목표지점 같은 상위 라우팅 판단(어느 분기로 갈지, 언제 유턴할지)은 이 모듈의
책임이 아니다 — Signal-Simulation이 이미 웨이포인트 순서를 알고 있으므로, 이 모듈은 "지금
눈앞 트랙에 잘 붙어서 따라가기"(차선 유지)만 담당한다. 화면에 여러 직선 차선/분기가 동시에
보이는 상황에서 로봇이 지금 타고 있는 트랙만 골라내는 것(select_own_track)과, 다른 차량이
트랙을 가리거나 색이 비슷해 오검출을 일으키는 걸 막는 것(obstacle_mask, 이 레포의 YOLO
Vehicle/Person 탐지 결과를 그대로 재사용)까지가 이 모듈의 책임이다.

트랙 색상 HSV 범위는 navi_factory.sdf의 track_* 마커 재질(ambient/diffuse 1 0.35 0 1)을
실제 Gazebo(ogre2) 렌더로 캡처해 측정한 값이다 — 그림자 진 면도 색상(H)·채도(S)는 거의
그대로 유지되고 명도(V)만 낮아지는 걸 확인해서, V는 넓게 H/S는 좁게 잡았다.

ROS 2로 넘기는 출력 인터페이스는 이 프로젝트 전체적으로 아직 미정(docu/decisions.md
ADR-005/010)이라, 이 모듈은 순수 인식 함수(이미지 → 오프셋/곡률)까지만 만들고 실제 노드
배선은 인터페이스가 정해진 뒤에 잇는다.
'''

import cv2
import numpy as np

TRACK_HSV_LOW = np.array([10, 120, 60])
TRACK_HSV_HIGH = np.array([28, 255, 255])

ROI_TOP_FRAC = 0.45  # 프레임 상단 이 비율까지는 트랙 탐색에서 제외(하늘/먼 배경)
NUM_BANDS = 6  # 곡선 피팅에 쓸 가로 스트립 개수(가까운 곳 → 먼 곳)
CURVE_DEGREE = 2
MIN_COMPONENT_AREA = 20
BRANCH_JUMP_THRESHOLD = 0.5  # 오프셋이 한 프레임 만에 이 이상 튀면 다른 분기로 오인한 것으로 봄

TrackEstimate = tuple[float, float, float]  # (offset, curvature, confidence)


def obstacle_mask(shape: tuple[int, int], boxes_xyxy: list[tuple[float, float, float, float]], pad: int = 6) -> np.ndarray:
    '''YOLO 탐지 박스(Vehicle/Person 등) 영역을 True로 표시한 마스크. 트랙 색상 마스크에서
    이 영역을 빼서, 트랙 위에 서 있는 다른 차량/사람 때문에 생기는 끊김·오검출을 막는다.'''
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for x1, y1, x2, y2 in boxes_xyxy:
        px1, py1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        px2, py2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        mask[py1:py2, px1:px2] = True
    return mask


def track_color_mask(frame_bgr: np.ndarray, obstacle_boxes: list[tuple] | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TRACK_HSV_LOW, TRACK_HSV_HIGH) > 0
    h = frame_bgr.shape[0]
    mask[: int(h * ROI_TOP_FRAC), :] = False
    if obstacle_boxes:
        mask &= ~obstacle_mask(frame_bgr.shape[:2], obstacle_boxes)
    return mask


def select_own_track(mask: np.ndarray) -> np.ndarray | None:
    '''화면에 여러 트랙 분기가 동시에 보여도, 로봇 바로 앞(화면 하단 중앙)에 맞닿은 연결
    성분 하나만 "지금 타고 있는 트랙"으로 고른다. 중심점이 아니라 "로봇에 가장 가까운 끝"을
    기준으로 골라야, 화면 저 멀리 지나가는 다른 차선과 헷갈리지 않는다.'''
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None

    h, w = mask.shape
    anchor = np.array([w / 2, h - 1])
    best_label, best_dist = None, None
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue
        ys, xs = np.where(labels == label)
        near_idx = ys.argmax()
        near_point = np.array([xs[near_idx], ys[near_idx]])
        dist = float(np.linalg.norm(near_point - anchor))
        if best_dist is None or dist < best_dist:
            best_dist, best_label = dist, label

    if best_label is None:
        return None
    return labels == best_label


def fit_curve(track_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]] | None:
    '''선택된 트랙 성분을 가까운 밴드 → 먼 밴드로 나눠 각 밴드의 x중심을 뽑고, 다항식으로
    피팅한다(x = f(y)). 코너에서 앞으로 얼마나 꺾이는지까지 반영하려면 직선(1차)만으로는
    부족해서 기본 2차를 쓴다.'''
    ys, xs = np.where(track_mask)
    if len(ys) < 10:
        return None

    y_min, y_max = ys.min(), ys.max()
    band_edges = np.linspace(y_min, y_max, NUM_BANDS + 1)
    points = []
    for i in range(NUM_BANDS):
        lo, hi = band_edges[i], band_edges[i + 1]
        sel = (ys >= lo) & (ys < hi if i < NUM_BANDS - 1 else ys <= hi)
        if sel.sum() < 3:
            continue
        points.append((float(xs[sel].mean()), float(ys[sel].mean())))

    if len(points) < 3:
        return None
    xs_pts = np.array([p[0] for p in points])
    ys_pts = np.array([p[1] for p in points])
    coeffs = np.polyfit(ys_pts, xs_pts, deg=min(CURVE_DEGREE, len(points) - 1))
    return coeffs, points


def estimate_track(
    frame_bgr: np.ndarray,
    obstacle_boxes: list[tuple] | None = None,
    previous: TrackEstimate | None = None,
) -> TrackEstimate | None:
    '''로봇 바로 앞 트랙의 (offset, curvature, confidence)를 추정한다.

    offset: 화면 하단(로봇 바로 앞)에서 트랙 중심이 화면 중앙 대비 얼마나 치우쳤는지,
        -1(왼쪽 끝) ~ 1(오른쪽 끝)로 정규화.
    curvature: 2차 다항식 계수(x=f(y))의 2차항 — 0에 가까우면 직진, 부호로 꺾이는 방향을 안다.
    confidence: 이번 프레임에서 트랙을 얼마나 확실하게 찾았는지(0~1). 다른 차량에 가려 일부만
        보이면 낮아진다 — 호출부(Signal-Simulation 조향 로직)가 낮은 confidence일 때 직전
        추정치를 유지할지 감속할지 판단하는 데 쓸 수 있다.
    previous: 직전 프레임의 추정치. 주어지면 (1) 오프셋이 갑자기 크게 튀는 경우(=다른 분기로
        잘못 물린 것으로 봄) 억제하고 (2) 프레임 간 값을 평활화한다.
    '''
    h, w = frame_bgr.shape[:2]
    mask = track_color_mask(frame_bgr, obstacle_boxes)
    own = select_own_track(mask)
    if own is None:
        return None

    fit = fit_curve(own)
    if fit is None:
        return None
    coeffs, _ = fit

    near_x = float(np.polyval(coeffs, h - 1))
    offset = (near_x - w / 2) / (w / 2)
    curvature = float(coeffs[0]) if len(coeffs) >= 3 else 0.0
    confidence = min(1.0, int(own.sum()) / (0.02 * h * w))

    if previous is not None:
        prev_offset, prev_curvature, _ = previous
        if abs(offset - prev_offset) > BRANCH_JUMP_THRESHOLD:
            offset = prev_offset + float(np.clip(offset - prev_offset, -0.15, 0.15))
            confidence *= 0.5
        offset = 0.6 * offset + 0.4 * prev_offset
        curvature = 0.6 * curvature + 0.4 * prev_curvature

    return offset, curvature, confidence
