# M.AX Project Brief

## 프로젝트 개요

Franka FR3 로봇 + 카메라 4대 + VLA 모델(pi0.5)을 통합하는 시스템.  
ROS2 ↔ ZMQ 브릿지 서버 + Streamlit 모니터링/제어 웹앱 구조.

```
[ROS2 Nodes] ←→ [CommServer (ZMQ Bridge)] ←→ [Streamlit Web App]
  domain_id=0       PUB :5570 (robot/gripper)
  domain_id=1       PUB :5571 (camera)
                    PULL :5590 (제어 명령)
                    PUB  :5591 (제어권 알림)
```

---

## 파일 구조

```
m.ax/
├── server.py                     # CommServer 진입점: python server.py
├── app.py                        # Streamlit 앱 진입점: streamlit run app.py
├── config/
│   ├── server_config.yaml        # 프로젝트 목록 (gt_kitting → proj_gt_kitting.yaml)
│   └── proj_gt_kitting.yaml      # gt_kitting 프로젝트 전체 설정
├── common/
│   ├── comm_server.py            # ROS ↔ ZMQ 브릿지 (DomainBridge + Arbiter)
│   ├── vla_server.py             # VLA 추론 서버 (별도)
│   └── utils/
│       ├── utils.py              # load_config (OmegaConf merge)
│       ├── ros_utils.py          # ROS↔ZMQ 헬퍼, msg type registry
│       └── ui_utils.py           # Streamlit UI 컴포넌트 모음
├── pages/
│   ├── monitor.py                # 실시간 모니터링 + 제어
│   ├── collection.py             # 데이터 수집 (placeholder)
│   └── inference.py              # 추론 실행 (placeholder)
└── docker/
    ├── Dockerfile.jetpack        # Jetson AGX용
    ├── Dockerfile.x86            # x86용
    ├── build_docker.sh
    └── container.sh
```

---

## 하드웨어 (gt_kitting 프로젝트)

| 장치 | 타입 | `ros_domain_id` | Subscribe | Publish |
|------|------|----------------|-----------|---------|
| Robot | Franka FR3 (172.16.0.2) | 0 | `/joint_states` (JointState) | `/gello/joint_states` (JointState) |
| Gripper | dxl_parallel_gripper | 0 | `/dxl_parallel_gripper/joint_states` (JointState) | `/gripper/gripper_client/target_gripper_width_percent` (Float32) |
| Camera 1 | RealSense D405 `wrist_front` | 1 | `/wrist/front/color/image_raw/compressed` | — |
| Camera 2 | RealSense D405 `wrist_rear` (rotate 180°) | 1 | `/wrist/rear/color/image_raw/compressed` | — |
| Camera 3 | Orbbec Gemini2 `front_view` | 1 | `/front_view/color/image_raw/compressed` | — |
| Camera 4 | Orbbec Gemini2 `side_view` | 1 | `/side_view/color/image_raw/compressed` | — |

Robot pose 프리셋: `home`, `kit` (7-DOF joint values)  
Gripper pose 프리셋: `open: [1.0]`, `close: [0.0]` (웹 UI에서 0~100% 변환)

---

## CommServer (`common/comm_server.py`)

### 구성 요소

**`DomainBridge`**  
- 단일 `ros_domain_id`에 대한 ROS↔ZMQ 브릿지
- ROS subscribe → ZMQ PUB: `[topic_bytes, pickled_msg]` 전송
- Arbiter 요청 시 ROS publish 수행 (publisher dict 보유)
- domain_id마다 독립적인 `rclpy.Context` + `SingleThreadedExecutor` 사용

**`Arbiter`**  
- PULL :5590 바인드, 클라이언트로부터 `[source, topic, pickled_msg]` 수신
- 동시에 하나의 source만 제어권 보유 (exclusive lock)
- `LOCK_TIMEOUT = 5.0`초 동안 메시지 없으면 제어권 자동 해제
- 제어권 변경 시 PUB :5591으로 `granted:<source>` / `revoked:<source>` 발행

**`CommServer`**  
- cfg 파싱 → domain_id별 DomainBridge 생성 + Arbiter 기동

### rclpy 필수 규칙
- 모든 `rclpy.init()` 호출에 `signal_handler_options=SignalHandlerOptions.NO` 필수
- 미적용 시 다중 signal handler 등록 → core dump (`terminate called without active exception`)

---

## ZMQ 포트 규칙

| 포트 | 소켓 타입 | 방향 | 용도 |
|------|-----------|------|------|
| `5570 + domain_id` | PUB (CommServer 바인드) / SUB (Web 연결) | ROS → Web | 토픽 모니터링 |
| `5590` | PULL (CommServer 바인드) / PUSH (Web 연결) | Web → CommServer | 제어 명령 |
| `5591` | PUB (CommServer 바인드) / SUB (Web 연결) | CommServer → Web | 제어권 granted/revoked 알림 |

제어 메시지 포맷: `[source_bytes, topic_bytes, pickle.dumps(ROS_msg)]`

---

## Streamlit 앱

### 앱 구조 (`app.py`)
- `@st.cache_resource`로 cfg 로드 후 `st.session_state.cfg`에 저장
- `st.navigation`에 named function 사용 필수 (lambda 사용 시 URL pathname 충돌)

```python
pg = st.navigation({
    "Monitor":   [st.Page(page_monitor,    title="Monitor",         icon="📷")],
    "Operation": [st.Page(page_collection, title="Data Collection", icon="💾"),
                  st.Page(page_inference,  title="Inference",       icon="🤖")],
})
```

### Monitor 페이지 루프 (`pages/monitor.py`)
```python
def show(cfg):
    ph_cameras = render_cameras(cfg)
    ph_robot   = render_robot(cfg)
    ph_gripper = render_gripper(cfg)
    ph_ctrl    = render_control(cfg)
    while True:
        update_cameras(ph_cameras)
        update_robot(ph_robot, cfg)
        update_gripper(ph_gripper, cfg)
        update_control_status(ph_ctrl)
        time.sleep(0.033)   # ~30fps
```

---

## UI Utils (`common/utils/ui_utils.py`)

### ZMQ 소켓 관리 규칙

1. **단일 Context 공유**: `_zmq_context()`를 `@st.cache_resource`로 등록
   - 소켓마다 `zmq.Context()` 별도 생성 시 → `Assertion failed: !_more` core dump
2. **소켓별 캐시 함수 분리**: `_camera_sub_socket`, `_robot_sub_socket`, `_gripper_sub_socket`
   - 하나로 통합 시 소켓 오염(cross-contamination) 발생

### 카메라 이미지 표시
- `st.image(numpy)` 금지 → `while True` 안에서 `MediaFileStorageError` 발생
- **해결**: base64 inline HTML 사용
  ```python
  _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
  b64 = base64.b64encode(buf).decode()
  placeholder.html(f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;">')
  ```

### 최신 프레임만 취득하는 패턴 (`_recv_latest`)
```python
# 소켓 옵션: RCVTIMEO=100ms, RCVHWM=1
result = None
try:
    while True:
        parts = sock.recv_multipart(zmq.NOBLOCK)
        try:
            topic = parts[0].decode("utf-8")   # pickle bytes decode 방어 필수
            result = (topic, parts[1])
        except UnicodeDecodeError:
            pass
except zmq.Again:
    pass
```

### 제어 위젯 상태 관리 규칙
- Slider에 `value=`와 `session_state[key]` 동시 사용 금지 → core dump
- **올바른 패턴**:
  ```python
  if skey not in st.session_state:
      st.session_state[skey] = default_val
  val = st.slider("label", min_value=..., max_value=..., key=skey)  # value= 없음
  ```
- Pose 버튼 → `st.session_state[key]` 직접 덮어쓰기로 slider 값 변경

### 제어권 리스너 (`_ctrl_listener_started`)
- `@st.cache_resource` 함수가 `None` 반환 시 cache 작동 불량 → 매번 재실행
- dict 반환으로 수정 (`{"started": True}`)하여 한 번만 실행 보장
- 배경 스레드 이름 `name="ctrl-listener"` 지정 (ScriptRunContext warning 무시)

---

## 설정 로딩 (`common/utils/utils.py`)

```python
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
# common/utils/utils.py 기준: .parent×3 = 프로젝트 루트
```
OmegaConf로 `server_config.yaml`과 `proj_*.yaml` merge 후 단일 cfg 반환.

---

## 주요 버그 수정 이력

| 증상 | 원인 | 해결 |
|------|------|------|
| `FileNotFoundError: proj_gt_kitting.yaml` | `CONFIG_DIR` 경로 오류 | `Path(__file__).parent.parent.parent / "config"` |
| core dump: `terminate called without active exception` | 다중 `rclpy.init()` → signal handler 충돌 | `SignalHandlerOptions.NO` |
| `StreamlitAPIException: Multiple Pages with URL pathname <lambda>` | `st.navigation`에 lambda 사용 | named function으로 교체 |
| `MediaFileStorageError` | `while True`에서 `st.image(numpy)` | base64 HTML `<img>`로 교체 |
| `ValueError: truth value of array is ambiguous` | `frame or _no_signal_img()` (numpy) | `frame if frame is not None else ...` |
| `UnicodeDecodeError: 0x80` | pickle bytes를 UTF-8 decode 시도 | try/except UnicodeDecodeError |
| Slider core dump | `value=` + `session_state[key]` 동시 설정 | `value=` 제거, session_state만 사용 |
| core dump: `Assertion failed: !_more (src/fq.cpp:80)` | 소켓마다 별도 `zmq.Context()` 생성 | 단일 `_zmq_context()` 공유 |
| `@st.cache_resource` 함수 매번 재실행 | `None` 반환 시 cache 미작동 | dict 반환으로 수정 |

---

## 실행 방법

```bash
# 터미널 1: CommServer (ROS↔ZMQ 브릿지)
python server.py --project gt_kitting

# 터미널 2: Streamlit 웹앱
streamlit run app.py
# → http://localhost:8501
```

---

## 미구현 항목

- `pages/collection.py` — 데이터 수집 기능 (현재 placeholder)
- `pages/inference.py` — 추론 실행 기능 (현재 placeholder)
