# ros2/src

M.AX 시스템의 ROS 2 워크스페이스. 카메라 드라이버 + 추론/제어 서버 + 웹 UI를 한 워크스페이스에서
빌드/실행한다.

워크스페이스 루트는 `ros2/`이며, `colcon build`는 `ros2/src/` 하위의 모든 패키지를 인식한다.
`max_web`은 `COLCON_IGNORE`가 있어 colcon이 건너뛰고, 별도 `npm` 빌드를 사용한다.

---

## 1. 패키지 개요

| 패키지 | 빌드 타입 | 역할 |
|---|---|---|
| [max_interfaces](max_interfaces/) | `ament_cmake` (rosidl) | msg / srv / action 인터페이스 정의 |
| [max_server](max_server/) | `ament_python` | 추론·제어 서버 노드 (`max_server`) + launch + config |
| [max_bringup](max_bringup/) | `ament_python` | 카메라 드라이버 + domain_bridge + max_server를 묶어 띄우는 launch 모음 |
| [max_web](max_web/) | `npm` (colcon 제외) | React + FastAPI 기반 웹 프론트/백엔드 — 자세한 내용은 [max_web/README.md](max_web/README.md) |

### 1.1 max_interfaces

ROS 2 인터페이스 정의만 들어 있는 패키지. `max_server`와 `max_web`이 공통으로 의존한다.

- **msg**
  - `InferenceStatus.msg` — 서버 상태 + 로드된 정책 + 기본값 + 프리셋 포즈 이름. `/inference/status`로 1Hz 발행
  - `RobotStatesPackage.msg` — 관측된 joint/pose/gripper 상태 + 마지막 발행한 명령. `/control/robot_state_package`로 30Hz 발행
- **srv**
  - `MoveToPose.srv` — `target`(robot/gripper) + `pose_location`(YAML 프리셋 키) + `representation_type`(joint/quat/rot6d)을 받아 프리셋 포즈 발행
  - `UnloadPolicy.srv` — 현재 정책 언로드
- **action**
  - `LoadPolicy.action` — `framework`/`policy`/`checkpoint`/`device`/`dtype`로 정책 로드 (피드백: `stage`/`detail`)
  - `RunInference.action` — `command`(START/STOP) + `task_instruction` + `representation_type`로 추론 루프 시작/정지

### 1.2 max_server

핵심 노드 [max_server/max_server.py](max_server/max_server/max_server.py). 다음 ROS 인터페이스를 노출한다:

- Action `/inference/load_policy` (`LoadPolicy`) — 정책 로드
- Service `/inference/unload_policy` (`UnloadPolicy`) — 정책 언로드
- Service `/control/move_to_pose` (`MoveToPose`) — YAML 프리셋 포즈 발행
- Action `/inference/run` (`RunInference`) — 추론 루프 START/STOP
- Topic `/inference/status` (`InferenceStatus`, 1Hz) — 서버 상태 브로드캐스트
- Topic `/control/robot_state_package` (`RobotStatesPackage`, 30Hz) — 텔레메트리 브로드캐스트

내부 모듈:

- `communication/communicator.py` — 그룹별 `ROS_DOMAIN_ID`마다 독립적인 rclpy Context/Node/Executor를 띄워 sub/pub을 처리. 카메라가 도메인 1, 로봇/추론이 도메인 0 같은 멀티-도메인 토폴로지를 지원
- `inference/inference_manager.py` — LeRobot 정책(pi0.5, SmolVLA 등) 로드/언로드/predict. ROS 의존성 없음
- `data_processing/data_converter.py` — 이미지/상태 전처리, 표현형(joint/quat/rot6d) 변환
- `task_manager/`, `task_planner/` — Phase 2 placeholder
- `utils/config_loader.py` — YAML `"role:topic"` 항목 파싱

설정 파일: [max_server/config/kitting_config.yaml](max_server/config/kitting_config.yaml)
- `inference.*` — fps, default 정책/체크포인트/태스크, 관측 리스트, `ros_domain_id`
- `robot.*` — joint 이름, 프리셋 포즈(home/kit), sub/pub 토픽, `ros_domain_id`
- `gripper.*` — 프리셋(open/close), sub/pub 토픽, `ros_domain_id`
- `camera.*` — 4채널(wrist_front/wrist_rear/front_view/side_view) sub 토픽 + 회전 각도, `ros_domain_id`

### 1.3 max_bringup

launch 파일 모음 (Python 코드는 없음, 모두 `share/`로 설치되는 launch + config).

- [`launch/bringup.launch.py`](max_bringup/launch/bringup.launch.py) — 통합 런치. `use_cameras:=true|false`, `use_domain_bridge:=true|false`로 토글. 내부에서 `camera.launch.py`와 `max_server.launch.py`를 include
- [`launch/camera.launch.py`](max_bringup/launch/camera.launch.py) — RealSense D405 ×2(`wrist_front`, `wrist_rear`) + D435 ×2(`front_view`, `side_view`)를 USB 경합 회피를 위해 0/2/4/6초 간격으로 순차 기동. `additional_env`로 각 노드에 `ROS_DOMAIN_ID`를 주입(기본 1)
- [`launch/camera_rs_ob.launch.py`](max_bringup/launch/camera_rs_ob.launch.py) — RealSense ×2 + Orbbec Gemini2 primary/secondary sync 조합 (별도 하드웨어 셋업용)
- [`launch/domain_bridge.launch.py`](max_bringup/launch/domain_bridge.launch.py) — `domain_bridge` 패키지를 [config/domain_config.yaml](max_bringup/config/domain_config.yaml)로 실행해 도메인 0 ↔ 1 사이에서 joint_states / current_pose / gripper 토픽을 양방향 포워딩

### 1.4 max_web

ROS 2 빌드에서는 제외되며(`COLCON_IGNORE`), `vite` 개발 서버(5173) 또는 정적 빌드(`dist/`)로 실행한다.
rosbridge(WebSocket, 9090)와 web_video_server(MJPEG, 8080)를 통해 ROS 측과 통신.
자세한 사용법은 [max_web/README.md](max_web/README.md) 참조.

---

## 2. 빌드

```bash
cd ros2

# colcon 빌드 (max_web 제외 자동 처리됨)
colcon build --symlink-install

# 환경 source
source install/setup.bash
```

사전 요구사항:

- ROS 2 Humble (또는 호환 배포판) + `rosidl_default_generators`, `action_msgs`
- `realsense2_camera` (camera launch)
- `orbbec_camera` (camera_rs_ob launch만 사용 시)
- `domain_bridge` (domain_bridge launch만 사용 시)
- `rosbridge_server`, `web_video_server` (max_server.launch.py가 함께 띄움)
- Python: `torch`, `numpy`, `opencv-python`, `lerobot` (정책 로드용)

---

## 3. 실행 방법

ROS_DOMAIN_ID는 기본적으로 카메라가 도메인 **1**, 추론/제어가 도메인 **0**으로 분리되어 있다.
`max_server`는 YAML의 `<group>.ros_domain_id`를 읽어 그룹마다 독립 Context를 만들기 때문에,
한 프로세스 안에서 두 도메인을 동시에 사용한다.

### 3.1 통합 실행 (권장)

카메라 + max_server + rosbridge + web_video_server를 한 번에:

```bash
ros2 launch max_bringup bringup.launch.py
```

옵션:

```bash
# 카메라 없이 서버만
ros2 launch max_bringup bringup.launch.py use_cameras:=false

# domain_bridge까지 함께 (카메라 토픽을 도메인 1 → 0으로 포워딩)
ros2 launch max_bringup bringup.launch.py use_domain_bridge:=true
```

### 3.2 개별 실행

```bash
# 카메라만 (기본 ROS_DOMAIN_ID=1)
ros2 launch max_bringup camera.launch.py
ros2 launch max_bringup camera.launch.py video_domain_id:=2

# RealSense + Orbbec 조합
ros2 launch max_bringup camera_rs_ob.launch.py cam_domain_id:=1

# domain_bridge 단독
ros2 launch max_bringup domain_bridge.launch.py
ros2 launch max_bringup domain_bridge.launch.py config_file:=/path/to/custom.yaml

# max_server + rosbridge + web_video_server
ros2 launch max_server max_server.launch.py

# 커스텀 설정 / 도메인 오버라이드
ros2 launch max_server max_server.launch.py \
    config_file:=/path/to/config.yaml \
    bridge_domain_id:=0 \
    video_domain_id:=1

# max_server 노드만 단독으로 (rosbridge/web_video 없이)
ros2 run max_server max_server --ros-args --params-file \
    $(ros2 pkg prefix max_server)/share/max_server/config/kitting_config.yaml
```

### 3.3 웹 UI 실행

별도 터미널에서:

```bash
cd ros2/src/max_web
npm install        # 최초 1회
npm run dev        # http://localhost:5173
```

브라우저에서 `http://localhost:5173` 접속 후 상단 ConnectionBar에서 rosbridge 호스트 입력 → 연결.
MJPEG 스트림은 `http://<host>:8080/stream?topic=...`을 자동으로 사용한다.

---

## 4. 빠른 점검

```bash
# 토픽/서비스/액션 확인 (ROS_DOMAIN_ID=0)
ros2 topic list
ros2 topic echo /inference/status
ros2 topic echo /control/robot_state_package
ros2 action list

# 카메라 토픽 확인 (ROS_DOMAIN_ID=1)
ROS_DOMAIN_ID=1 ros2 topic list | grep observation
ROS_DOMAIN_ID=1 ros2 topic hz /observation/front_view/color/image_raw/compressed

# 프리셋 포즈 호출 예시
ros2 service call /control/move_to_pose max_interfaces/srv/MoveToPose \
    "{target: robot, pose_location: home, representation_type: rot6d}"
```
