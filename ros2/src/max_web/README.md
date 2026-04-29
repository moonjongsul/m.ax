# max_web

M.AX 시스템의 웹 프론트엔드. ROS 2 위에서 동작하는 `max_server`를 브라우저에서 제어하기 위한
React 기반 SPA(Single Page Application)이며, LeRobot v3.0 데이터셋을 시각화·트리밍하기 위한
별도의 FastAPI 백엔드(`backend/lerobot_editor/server.py`)를 함께 포함한다.

빌드 시스템은 ROS 2의 `colcon`이 아닌 `vite`를 사용한다 — 패키지 루트의 `COLCON_IGNORE`가
이를 명시한다. 즉 `colcon build`는 이 패키지를 건너뛴다.

---

## 1. 디렉토리 구조

```
max_web/
├── COLCON_IGNORE             # colcon이 이 패키지를 빌드하지 않도록 제외
├── package.json              # npm 의존성 / 스크립트
├── vite.config.js            # 개발 서버 (5173) + /api → :8765 프록시
├── tailwind.config.js        # Tailwind 콘텐츠 경로
├── postcss.config.js         # tailwind + autoprefixer
├── index.html                # SPA entrypoint
├── src/
│   ├── main.jsx              # React 18+ root mount + Redux Provider
│   ├── App.jsx               # 상단 ConnectionBar + 페이지 스위처(Inference / LeRobot)
│   ├── index.css             # Tailwind directives
│   ├── store/store.js        # Redux store: ros / inference / telemetry / lerobot
│   ├── pages/
│   │   ├── InferencePage.jsx        # 카메라 + 텔레메트리 + 추론 컨트롤
│   │   └── LeRobotEditorPage.jsx    # 데이터셋 에디터
│   ├── hooks/                # rosbridge 래퍼 훅 (singleton 기반)
│   │   ├── useRosConnection.js
│   │   ├── useRosTopicSubscription.js
│   │   ├── useRosServiceCaller.js
│   │   └── useRosActionClient.js
│   ├── features/             # Redux slice 단위로 도메인 분리
│   │   ├── ros/rosSlice.js              # 연결 상태, host/포트
│   │   ├── inference/inferenceSlice.js  # /inference/status 미러
│   │   ├── telemetry/telemetrySlice.js  # /control/robot_state_package 미러
│   │   └── lerobot/                     # 에디터 thunk + slice + fetch helper
│   └── components/
│       ├── CameraView.jsx           # web_video_server MJPEG 스트림 4분할
│       ├── InferenceStatus.jsx      # 서버 상태 / 정책 / 태스크 표시
│       ├── PolicyLoader.jsx         # /inference/load_policy 액션
│       ├── InferenceControl.jsx     # /inference/run 액션 (Start/Stop)
│       ├── PosePanel.jsx            # /control/move_to_pose 서비스
│       ├── RobotStatesPanel.jsx     # 30 Hz 텔레메트리 차트 (uPlot)
│       └── lerobot/                 # 에피소드 리스트, 비디오 그리드, 타임라인 등
└── backend/
    └── lerobot_editor/server.py     # FastAPI 백엔드 (LeRobot v3.0 데이터셋 IO)
```

---

## 2. 빌드 및 실행

### 2.1 사전 요구사항

- **Node.js 18+** / npm
- **Python 3.10+** (LeRobot Editor 백엔드 사용 시)
- **ROS 2 Humble**(또는 호환 버전) 환경에 다음 패키지가 설치되어 있어야 한다:
  - `rosbridge_server` (websocket → ROS bridge, 포트 9090)
  - `web_video_server` (이미지 토픽 → MJPEG HTTP, 포트 8080)
- 빌드된 `max_interfaces` (msg/srv/action 정의 — 웹 UI는 string 기반으로 호출하므로 직접 의존하지는 않지만 `max_server`가 사용)

### 2.2 프론트엔드 설치 및 실행

```bash
cd ros2/src/max_web

# 1) 의존성 설치
npm install

# 2-A) 개발 서버 (HMR, 5173 포트)
npm run dev

# 2-B) 프로덕션 빌드 → dist/
npm run build

# 2-C) 빌드 결과 미리보기 (4173 포트)
npm run preview
```

`vite.config.js`는 `host: true`로 외부 접속을 허용하고, `/api/*` 요청을 `http://127.0.0.1:8765`
(LeRobot 백엔드)로 프록시한다.

### 2.3 ROS / 비디오 브릿지 실행

웹 UI는 `max_server` launch 파일이 같이 띄우는 `rosbridge_server`(9090)와 `web_video_server`(8080)에
의존한다. 별도 launch가 아니라 `max_server.launch.py` 한 번에 같이 올라간다:

```bash
# 패키지 빌드 후 환경 source
source install/setup.bash

ros2 launch max_server max_server.launch.py
# 옵션:
#   bridge_domain_id:=0   # rosbridge가 사용할 ROS_DOMAIN_ID (= inference.ros_domain_id)
#   video_domain_id:=1    # web_video_server가 사용할 ROS_DOMAIN_ID (= camera.ros_domain_id)
#   config_file:=...      # 기본 kitting_config.yaml
```

내부적으로 `max_server` 노드, `rosbridge_websocket`, `web_video_server` 세 노드가 같이 시작된다.
`rosbridge`와 `web_video_server`는 각각 다른 `ROS_DOMAIN_ID`로 실행될 수 있다 — `max_server`가
inference / robot / gripper / camera 별로 도메인을 분리할 수 있도록 설계됐기 때문이다.

### 2.4 LeRobot Editor 백엔드 실행 (선택)

LeRobot Editor 페이지를 사용하려면 별도의 FastAPI 서버를 8765 포트로 띄워야 한다:

```bash
cd ros2/src/max_web/backend/lerobot_editor

# 의존성: fastapi, uvicorn, pyarrow, numpy, pydantic, lerobot
python server.py
# 또는
uvicorn server:app --host 0.0.0.0 --port 8765 --reload
```

Vite dev 서버가 켜져 있으면 `/api/*` 요청은 자동으로 8765로 프록시된다.
프로덕션 배포 시에는 nginx 등 프론트 서버에서 동일하게 `/api`를 프록시 매핑해야 한다.

### 2.5 접속

- 개발: `http://<host>:5173`
- 프로덕션 미리보기: `http://<host>:4173`

브라우저는 자동으로 `window.location.hostname`을 사용해 `ws://<host>:9090`(rosbridge)와
`http://<host>:8080`(web_video_server)에 접속한다 (`rosSlice.js`의 selector 참고).

---

## 3. 통신 아키텍처

### 3.1 전체 토폴로지

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Browser (max_web)                         │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐     │
│  │  React UI    │  │  roslib.js   │  │  fetch() → /api/*       │     │
│  │  + Redux     │  │  (WebSocket) │  │  (LeRobot Editor 전용)  │     │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬────────────┘     │
│         │                 │                       │                  │
│         │                 │ ws://host:9090        │  http://host:5173│
│         │                 │                       │  /api → :8765    │
│         │ <img src=…>     │                       │                  │
│         │ http://host:8080/stream?topic=...                          │
└─────────┼─────────────────┼───────────────────────┼──────────────────┘
          │                 │                       │
          ▼                 ▼                       ▼
   ┌────────────┐  ┌─────────────────┐   ┌──────────────────────────┐
   │ web_video_ │  │ rosbridge_      │   │ FastAPI                  │
   │ server     │  │ websocket       │   │ (lerobot_editor/server.py)│
   │ (port 8080)│  │ (port 9090)     │   │ (port 8765)              │
   └─────┬──────┘  └─────────┬───────┘   └────────────┬─────────────┘
         │ ROS topic         │ pub/sub/srv/action     │ filesystem
         │ /observation/...  │                        │ (parquet, mp4)
         ▼                   ▼                        ▼
   ┌────────────────────────────────────┐   ┌──────────────────────┐
   │       ROS 2 (max_server)           │   │  LeRobot v3.0 dataset│
   │  - Inference loop                  │   │  (read-only or copy) │
   │  - /inference/status   topic       │   └──────────────────────┘
   │  - /control/robot_state_package    │
   │  - /inference/load_policy action   │
   │  - /inference/run         action   │
   │  - /inference/unload_policy srv    │
   │  - /control/move_to_pose  srv      │
   └────────────────────────────────────┘
```

브라우저가 백엔드와 통신하는 채널은 **3개**로 분리되어 있다:

| 채널 | 프로토콜 | 포트 | 용도 |
|---|---|---|---|
| **rosbridge** | WebSocket (JSON) | 9090 | 모든 ROS pub/sub/service/action 호출 |
| **web_video_server** | HTTP MJPEG | 8080 | 카메라 이미지 토픽을 `<img src>`로 직접 임베드 |
| **FastAPI (LeRobot)** | HTTP REST | 8765 | LeRobot 데이터셋 메타/프레임/비디오 readout, export 잡 |

### 3.2 ROS ↔ Web (rosbridge 경로)

`useRosConnection` 훅이 앱 마운트 시 한 번 `ROSLIB.Ros`를 생성해 모듈 스코프 싱글턴으로 보관한다.
이후 모든 hook (`useRosTopicSubscription`, `useRosServiceCaller`, `useRosActionClient`)은 `getRos()`로
이 싱글턴을 가져와 토픽/서비스/액션 객체를 만든다.

연결 상태(`connected`, `connecting`, `error`)와 host/포트(`host`, `rosbridgePort`, `videoPort`)는
`rosSlice`에 저장되어 UI 상단의 `ConnectionBar`와 카메라 URL 셀렉터(`selectVideoBaseUrl`)에서 사용된다.

#### 구독 (Topic)

| 토픽 | 메시지 타입 | 구독자 | 용도 |
|---|---|---|---|
| `/inference/status` | `max_interfaces/msg/InferenceStatus` (1 Hz) | `InferencePage` → `inferenceSlice.setStatus` | 서버 상태, 로드된 정책, 태스크, 폼 기본값(YAML), 프리셋 포즈 이름 |
| `/control/robot_state_package` | `max_interfaces/msg/RobotStatesPackage` (30 Hz) | `RobotStatesPanel` | 조인트/포즈/그리퍼 state·command, 차트 + 숫자 표 |

`RobotStatesPanel`은 30 Hz 메시지를 그대로 Redux에 디스패치하면 리렌더 비용이 크기 때문에:
- **링 버퍼(BUFFER_SECONDS × NOMINAL_HZ)** 에 직접 push하고
- **uPlot**이 `setInterval(50ms)`마다 데이터를 다시 그린다 (20 Hz draw)
- Redux 디스패치는 **100 ms 쓰로틀** (10 Hz)로 줄여 숫자 표만 갱신

이 분리 덕분에 차트 갱신이 React 렌더 사이클과 독립적이다.

#### 서비스 (Service)

| 서비스 | 타입 | 호출자 |
|---|---|---|
| `/inference/unload_policy` | `max_interfaces/srv/UnloadPolicy` | `PolicyLoader` |
| `/control/move_to_pose` | `max_interfaces/srv/MoveToPose` | `PosePanel` |

`useRosServiceCaller().call(name, type, payload, timeoutMs=120000)`이 Promise를 반환한다.
타임아웃 처리, 에러 변환을 한 곳에서 담당.

#### 액션 (Action)

| 액션 | 타입 | 호출자 | 라이프사이클 |
|---|---|---|---|
| `/inference/load_policy` | `max_interfaces/action/LoadPolicy` | `PolicyLoader` | feedback: stage/detail (예: `loading: …`), result: success/message |
| `/inference/run` | `max_interfaces/action/RunInference` | `InferenceControl` | feedback: step/last_action, cancel = Stop 버튼 |

`useRosActionClient`는 `roslib` 2.x의 ROS 2 네이티브 Action API(`ROSLIB.Action`)를 사용한다.
`sendGoal(goal, onResult, onFeedback, onFailed)`의 반환 ID를 ref에 저장해두고
`cancelGoal(id)`로 취소한다.

### 3.3 카메라 (HTTP MJPEG 경로)

ROS 이미지 토픽을 rosbridge로 직렬화하면 base64 + JSON 인코딩으로 트래픽이 폭증한다.
대신 `web_video_server`가 토픽을 직접 MJPEG으로 변환해 HTTP로 노출한다.
브라우저는 평범한 `<img>` 태그로 이를 받는다:

```jsx
<img src={`${base}/stream?topic=${topic}&type=mjpeg&quality=60`} />
```

`base`는 `http://<window.location.hostname>:8080`. 4개 카메라 토픽은 `CameraView.jsx`에 하드코딩되어 있다.

### 3.4 LeRobot Editor (REST 경로)

`LeRobotEditorPage`는 ROS와 무관하게 로컬 디스크의 LeRobot v3.0 데이터셋을 읽고 트리밍 후
재export하는 기능이다. FastAPI 서버가 다음 엔드포인트를 노출한다:

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/api/dataset/load` | 데이터셋 루트 경로를 받아 `meta/info.json`, `meta/episodes/*.parquet` 로드 |
| `GET` | `/api/dataset/episodes` | 에피소드 인덱스/길이/태스크 목록 |
| `GET` | `/api/dataset/episodes/{idx}/frames` | 에피소드의 action/state/timestamp 배열 + 비디오 메타 |
| `GET` | `/api/dataset/video?...` | mp4 파일을 HTTP Range로 서빙 (`<video>` 태그 시킹 지원) |
| `POST` | `/api/dataset/export` | 에피소드별 trim 범위로 새 데이터셋 export (백그라운드 스레드) |
| `GET` | `/api/dataset/export/status` | export 진행 상황 폴링 |

프론트엔드는 `features/lerobot/api.js`(thin fetch 래퍼) → `lerobotSlice` thunk로 호출한다.
모든 요청은 **상대 경로 `/api/...`** 로 보내고, 개발 환경에서는 Vite proxy가 8765로 라우팅한다.

---

## 4. 상태 관리 (Redux)

`@reduxjs/toolkit`의 `configureStore` + 4개 slice:

| Slice | 책임 | 주요 액션 |
|---|---|---|
| `ros` | rosbridge 연결 상태, host/포트 | `setHost`, `setConnected`, `setError` |
| `inference` | `/inference/status` 미러 + 폼 상태 | `setStatus`, `setFormField`, `resetFormToDefaults`, `setFeedback` |
| `telemetry` | `/control/robot_state_package` 미러 (10 Hz throttle) | `setRobotStates` |
| `lerobot` | 데이터셋/에피소드/플레이백/트림/export | `loadDatasetThunk`, `selectEpisodeThunk`, `startExportThunk`, `setPlayhead`, `setTrimStart/End` |

`inferenceSlice`의 폼은 **첫 status 수신 시점에 한 번** YAML 기본값(`default_*`)으로 초기화되며,
이후에는 사용자 입력이 우선이다. `formInitialized` 플래그로 이 동작을 제어한다.

`middleware.serializableCheck = false` — telemetry와 lerobot에서 timestamp/큰 배열을 직접 다루기
때문에 Redux Toolkit의 기본 직렬화 검사를 끈다.

---

## 5. 사용 라이브러리

### 프론트엔드 (`package.json`)

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| `react`, `react-dom` | ^19.1.0 | UI 프레임워크 |
| `@reduxjs/toolkit` | ^2.2.7 | 상태 관리 (slice + thunk) |
| `react-redux` | ^9.1.2 | React ↔ Redux 바인딩 |
| `roslib` | ^2.1.0 | rosbridge WebSocket 클라이언트 (ROS 2 네이티브 Action API 포함) |
| `uplot` | ^1.6.32 | 30 Hz 텔레메트리 라인 차트 (canvas, 가벼움) |
| `vite` | ^5.4.10 | 번들러 / dev 서버 |
| `@vitejs/plugin-react` | ^4.3.1 | React fast-refresh |
| `tailwindcss` | ^3.4.10 | 유틸리티 CSS |
| `postcss`, `autoprefixer` | — | Tailwind 빌드 파이프라인 |

### LeRobot Editor 백엔드

| 라이브러리 | 용도 |
|---|---|
| `fastapi` + `uvicorn` | HTTP API 서버 |
| `pydantic` | 요청 모델 검증 |
| `pyarrow` | LeRobot v3.0의 parquet IO |
| `numpy` | action/state 배열 처리 |
| `lerobot` (export 시점에만 import) | `LeRobotDataset.create`, `decode_video_frames` |

---

## 6. 페이지 구성

### 6.1 Inference Page

3-column 레이아웃:

- **좌측 (2/3 width)**
  - `CameraView`: 4개 카메라 MJPEG 스트림
  - `RobotStatesPanel`: joint/pose/gripper의 state·command 숫자 표 + 라이브 차트 4개
- **우측 (1/3 width)**
  - `InferenceStatus`: 서버 상태 dot + 현재 정책/태스크/스텝/마지막 액션
  - `PolicyLoader`: 프레임워크/정책/체크포인트/디바이스/dtype 입력 → Load/Unload
  - `InferenceControl`: 태스크 instruction + representation_type → Start/Stop
  - `PosePanel`: YAML에 정의된 프리셋 포즈를 robot/gripper별로 호출

### 6.2 LeRobot Editor Page

- 상단: `DatasetPathInput` — 데이터셋 루트 경로 입력 → `/api/dataset/load`
- 좌측: `EpisodeList` — 에피소드 목록 (선택 시 `/api/dataset/episodes/{idx}/frames`)
- 우측: `EpisodeViewer` (비디오 그리드 + uPlot 시계열 + 현재 프레임 값) + `TimelineBar` (재생/트림)

---

## 7. 트러블슈팅

- **"disconnected @ host:9090"**: `rosbridge_server`가 떠 있지 않거나 다른 도메인에 있다.
  `max_server.launch.py`의 `bridge_domain_id`가 `inference.ros_domain_id`와 같은지 확인.
- **카메라가 검정 화면**: `web_video_server`가 카메라 도메인(`camera.ros_domain_id`)에서 토픽을
  보지 못한다. `video_domain_id` 인자를 맞춰서 다시 launch.
- **Inference 시작이 거부됨**: `inferenceSlice`의 `serverState`가 `ready`여야 한다. 정책 미로드 시 `idle`.
- **LeRobot 페이지 500/CORS**: 백엔드 8765가 안 떠 있거나, 프록시 없이 절대 URL로 접근한 경우.
  반드시 `npm run dev` 또는 `npm run preview` 경유 (Vite proxy 사용).
- **차트가 빈칸**: `/control/robot_state_package`가 30 Hz로 발행되는지 (`max_server`의 telemetry 타이머
  `telemetry.robot_states_rate_hz` 파라미터) 확인.
