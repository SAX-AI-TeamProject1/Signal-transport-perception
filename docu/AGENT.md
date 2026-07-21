# Signal-transport-perception

## 목적

라이다가 탑재된 운송 로봇(Signal-Simulation에서 시뮬레이션)이 라이다로 충돌 가능성을 감지했을 때,
차량 탑재 카메라 영상을 YOLO로 분류해서 충돌 대상이 **사람 / 페트병 / 상자** 중 무엇인지 구분한다.
그 분류 결과로 "정지"(사람) 또는 "회피 이동"(사물) 중 어떤 반응을 할지가 결정된다.

## 다른 레포와의 관계

- **Signal-Vision**: 고정 카메라로 신호수(信号手)의 손동작을 인식해 정지/좌회전/우회전 명령을 만드는 별도 파이프라인. 카메라 종류·트리거 방식·개발 단계가 달라서 이 레포와는 분리되어 있음.
- **Signal-Simulation**: Gazebo 시뮬레이션 + ROS 2 통합을 담당. 라이다 충돌 감지와 로봇 제어(정지/회피 동작)는 그쪽에서 하고, 카메라 기반 이미지 처리는 절대 그 레포에 넣지 않기로 정함(Signal-Simulation의 CLAUDE.md에 명시된 경계). 이 레포는 분류 결과를 ROS 2 토픽/메시지로 넘겨주는 쪽.
- 출력 인터페이스(토픽 이름, 메시지 타입, 좌표계 등)는 아직 미정 — Signal-Simulation과 연동하는 코드를 짤 때 확정 필요.

## 개발 환경

- Python 3.12, 레포 루트 `.venv` (Signal-Vision / Signal-Simulation과 동일 버전 컨벤션)
- 의존성은 `requirements.txt`(ultralytics, opencv-python, numpy)
- VS Code 태스크(`.vscode/tasks.json`, macOS/Linux/Windows 호환):
  - **1. 패키지 일괄 설치** — `.venv` 생성(없으면) + `requirements.txt` 설치
  - **2. 날짜 자동찍기** — `tools/update_date_header.py`로 모든 `.py` 파일의 `# Last updated:` 헤더를 오늘 날짜로 일괄 갱신

## 현재 상태

폴더 구조(`asset/`, `data/`, `src/`)만 있고 코드는 아직 없음. 모델 학습/추론 코드는 팀원이 직접 작성 예정.
