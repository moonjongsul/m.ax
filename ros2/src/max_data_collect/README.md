# max_data_collect

수집 → 병합 → 라벨링 → 변환 → 업로드.

```
recorder_node  →  merge_raw_episode.py  →  max_data_editor  →  convert_to_lerobot.py  →  upload_to_hub.py
   원본 수집          여러 세션 합치기         구간/점수 라벨링        LeRobot 포맷            Hub 업로드
```

## 빌드

```bash
cd /workspace/m.ax/ros2
colcon build --packages-select max_data_collect
source install/setup.bash
```

---

## 1. 녹화 — `max_data_collect`

```bash
# launch (권장)
ros2 launch max_data_collect max_data_collect.launch.py

# 다른 config 파일로
ros2 launch max_data_collect max_data_collect.launch.py \
  config_file:=/path/to/my_config.yaml

# 노드 직접 실행
ros2 run max_data_collect max_data_collect \
  --ros-args --params-file config/demo_data_collect_config.yaml
```

설정: `config/demo_data_collect_config.yaml`

### 녹화 제어

`0=START  1=STOP  2=SAVE  3=DELETE`

```bash
ros2 service call /data_collection/record_command \
  max_interfaces/srv/RecordCommand "{command: 0}"    # 시작
ros2 service call /data_collection/record_command \
  max_interfaces/srv/RecordCommand "{command: 2}"    # 저장 (RECORDING에서 바로 호출 가능)
ros2 service call /data_collection/record_command \
  max_interfaces/srv/RecordCommand "{command: 3}"    # 폐기 / IDLE에서는 마지막 저장분 취소

# 상태 확인
ros2 topic echo /data_collection/recorder_status
```

---

## 2. 병합 — `merge_raw_episode.py`

여러 수집 세션을 에피소드 번호 재부여해서 하나로 합칩니다.
**합칠 대상은 스크립트 상단 `merge_target` 리스트를 직접 편집** ([scripts/merge_raw_episode.py:55](scripts/merge_raw_episode.py#L55)).

```bash
cd /workspace/m.ax

python3 ros2/src/max_data_collect/scripts/merge_raw_episode.py --dry-run
python3 ros2/src/max_data_collect/scripts/merge_raw_episode.py --out kitting_merged
```

| 인자 | 설명 |
|---|---|
| `--out NAME` | 출력 경로/이름 (기본 `kitting_dataset_xarm7_merged`) |
| `--dry-run` | 계획만 출력 |
| `--copy` | 하드링크 대신 복사 (원본과 완전 분리) |
| `--move` | 원본에서 이동 (원본 소모) |
| `--append` | 기존 출력에 추가 |
| `--force` | hz/robot/cameras 불일치해도 강행 |

---

## 3. 라벨링 — `max_data_editor`

구간(subtask) 분할, 프롬프트, 점수를 붙입니다. 결과는 각 에피소드의 `edits.json`.

```bash
ros2 run max_data_collect max_data_editor
ros2 run max_data_collect max_data_editor --root datasets/xarm7 --port 8020
```

브라우저: `http://localhost:8020`

| 인자 | 설명 |
|---|---|
| `--config PATH` | 기본 `config/editor_config.yaml` |
| `--root DIR` | 데이터셋 루트 |
| `--host` / `--port` | 기본 `0.0.0.0:8020` |

---

## 4. LeRobot 변환 — `convert_to_lerobot.py`

```bash
cd /workspace/m.ax

# 계획 점검 (feature 폭·라벨 커버리지 확인, 실제 변환과 동일하게 검증)
python3 ros2/src/max_data_collect/scripts/convert_to_lerobot.py --dry-run

# 변환
python3 ros2/src/max_data_collect/scripts/convert_to_lerobot.py \
  --out datasets/lerobot/xarm7_kitting_260923
```

설정: `config/data_convert_lerobot_config.yaml`

| 인자 | 설명 |
|---|---|
| `--config PATH` | 기본 `config/data_convert_lerobot_config.yaml` |
| `--out DIR` | 출력 디렉터리 (`--dry-run` 아니면 필수) |
| `--limit N` | 앞 N개만 변환 |
| `--dry-run` | 계획만 출력 |
| `--overwrite` | 기존 출력 교체 |

> `names` 길이가 실제 데이터 폭과 다르면 `--dry-run`에서 즉시 실패합니다. 변환 전 항상 먼저 돌리세요.

---

## 5. Hub 업로드 — `upload_to_hub.py`

```bash
cd /workspace/m.ax
hf auth login          # 또는 export HF_TOKEN=...

# 점검
python3 ros2/src/max_data_collect/scripts/upload_to_hub.py \
  --root datasets/lerobot/xarm7_kitting_260923 \
  --repo-id <user>/<name> --dry-run

# 업로드 (기본 private)
python3 ros2/src/max_data_collect/scripts/upload_to_hub.py \
  --root datasets/lerobot/xarm7_kitting_260923 \
  --repo-id <user>/<name>
```
```
cd /workspace/m.ax

# 1) 먼저 dry-run으로 점검 (업로드 안 함)
python3 ros2/src/max_data_collect/scripts/upload_to_hub.py \
  --root datasets/lerobot/xarm7_kitting_260923 \
  --repo-id moonjongsul/xarm7-kitting-260923 \
  --dry-run

# 2) 이상 없으면 실제 업로드 (기본 private)
python3 ros2/src/max_data_collect/scripts/upload_to_hub.py \
  --root datasets/lerobot/xarm7_kitting_260923 \
  --repo-id moonjongsul/xarm7-kitting-260923

```

| 인자 | 설명 |
|---|---|
| `--root DIR` | 변환된 데이터셋 (필수) |
| `--repo-id user/name` | Hub 저장소 (필수) |
| `--dry-run` | 점검만 |
| `--public` | 공개 (`public` 입력 확인 필요, 기본 private) |
| `--branch NAME` | 대상 브랜치 |
| `--tags ...` | 기본 `robotics xarm7 manipulation` |
| `--license` | 기본 `apache-2.0` |
| `--no-videos` | 메타+parquet만 |
| `--no-large-folder` | 재개 불가 업로더 사용 |
| `--force` | subtask 검사 실패해도 강행 |

> 토큰은 인자로 받지 않습니다. `hf auth login` 또는 `HF_TOKEN` 환경변수를 쓰세요.

---

## 설정 파일

| 파일 | 대상 |
|---|---|
| `config/demo_data_collect_config.yaml` | recorder_node (ROS 파라미터) |
| `config/editor_config.yaml` | max_data_editor (일반 YAML) |
| `config/data_convert_lerobot_config.yaml` | convert_to_lerobot.py |
