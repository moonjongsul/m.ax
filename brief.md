# M.AX Project Brief

## 프로젝트 개요
Franka FR3 로봇 + 카메라 + VLA 모델(pi0.5)을 통합 제어하는 시스템.
Streamlit 웹 UI + 단일 백엔드 서버 구조.

---

## 디렉토리 구조 (현재)
```
m.ax/
├── app.py                          # Streamlit UI 진입점 (거의 비어있음)
├── server.py                       # 백엔드 서버 진입점 (작성 중)
├── config/
│   ├── server_config.yaml          # 프로젝트 목록 (gt_kitting → proj_gt_kitting.yaml)
│   ├── ui_config.yaml              # UI 설정
│   └── proj_gt_kitting.yaml        # 프로젝트별 env/robot/camera/model 전체 설정
├── docker/
│   ├── Dockerfile.jetpack          # Jetson AGX용 (ROS2 Jazzy + lerobot + streamlit)
│   ├── docker-compose.yml          # runtime: nvidia, M_AX_ROOT 볼륨 마운트
│   ├── build_docker.sh             # m.ax/ 루트를 빌드 컨텍스트로 사용
│   └── container.sh                # create/start/enter/stop 등 컨테이너 관리
├── thirdparty/
│   └── lerobot/                    # git submodule (moonjongsul/lerobot)
└── gt_kitting/
    └── src/
```

---

## 아키텍처 결정사항

### 서버 구조
- **단일 server.py 프로세스** + 내부 멀티스레드 (카메라/로봇/모델 각각 스레드)
- 이유: 3개가 항상 같이 쓰이므로 별도 프로세스 불필요
- **서버 내부 통신**: `threading.Queue` / `dict` (직렬화 없이 메모리 공유)
- **UI ↔ 서버 통신**: ZMQ (카메라 프레임은 JPEG 압축 후 전송 ~50KB/frame)

### 실행 방식
```bash
python server.py --project gt_kitting   # 터미널 1
streamlit run app.py                    # 터미널 2
```

### Config 로딩 (server.py)
- `server_config.yaml` → 프로젝트명으로 `proj_*.yaml` 경로 조회
- OmegaConf로 두 파일 merge → `cfg.env.robot`, `cfg.env.camera`, `cfg.model` 등 접근
- **현재 상태**: `load_config()` 작성 완료, `omegaconf` 설치 필요

---

## 하드웨어 (gt_kitting 프로젝트)
- **로봇**: Franka FR3 (IP: 172.16.0.2), ROS2 통신
- **그리퍼**: dxl_parallel_gripper, ROS2 통신
- **카메라 4대**:
  - camera_1,2: RealSense D405 (848×480), /dev/video0,1
  - camera_3,4: Orbbec Gemini2 (1280×720), /dev/video2,3
- **모델**: VLA pi0.5

---

## Docker 환경
- **이미지**: `moonjongsul/max:cuda-13.2.0-ubuntu24.04-jetpack`
- **베이스**: `nvcr.io/nvidia/pytorch:25.12-py3` (Jetson SBSA)
- **ROS2**: Jazzy
- **유저**: `user` (UID/GID는 빌드 arg로 호스트와 맞춤)
- **볼륨**: 호스트 `m.ax/` → 컨테이너 `/workspace/m.ax` (실시간 연동)
- **pip**: `PIP_BREAK_SYSTEM_PACKAGES=1` 환경변수로 전역 설정
- **설치 패키지**: lerobot (editable), pyzmq, msgpack, streamlit, torchcodec 등

### 빌드/실행
```bash
cd ~/workspace/m.ax
./docker/build_docker.sh       # 이미지 빌드
./docker/container.sh create   # 컨테이너 생성 및 진입
./docker/container.sh enter    # 재진입
```

---

## 다음 작업
1. 컨테이너 안에서 `pip install omegaconf` 후 `server.py` 동작 확인
2. `server.py`에 CameraManager / RobotManager / ModelManager 스레드 추가
3. ZMQ 소켓 바인딩 (PUB for camera, REP for robot/model)
4. `app.py` Streamlit UI 및 `ui/` 페이지 구성
