# M.AX Dataset Editor

VLA 학습 데이터셋 생성을 위한 에피소드 뷰어 / 라벨링 도구.

`max_data_collect` 가 수집한 데이터셋을 편집하므로 이 패키지에 함께 두지만,
**ROS 노드가 아닙니다** — 독립 실행되는 FastAPI 웹 앱입니다.

## 실행

```bash
# 빌드된 워크스페이스에서
ros2 run max_data_collect max_data_editor

# 소스에서 직접 (ros2/src/max_data_collect 에서)
python3 -m max_data_collect.editor.server

# 다른 설정 파일로
python3 -m max_data_collect.editor.server --config /path/to/editor_config.yaml
```

## 설정 — `config/editor_config.yaml`

데이터셋 경로, 포트, 영상 프록시, 플롯 채널, 자동 분할 임계값, 라벨 사전을
모두 여기서 지정합니다. 우선순위는 **커맨드라인 플래그 > YAML > 내장 기본값**
이고, 오타나 잘못된 값은 **시작 시점에** 구체적인 메시지와 함께 거부됩니다.

| 섹션 | 내용 |
|---|---|
| `dataset` | `root` (데이터셋 루트, 상대경로는 리포 기준), `default_dataset` |
| `server` | `host`, `port` |
| `video` | `proxy`(auto/never/always), `playable_codecs`, 인코더·품질, 캐시 폴더명 |
| `timeline` | `max_points`, 플롯할 `channels` (key/label/reduce) |
| `autosegment` | gripper 키, 임계값, trim 여유 프레임, 최소 구간 길이, 경계 오프셋(`close_lead_sec`/`open_lag_sec`) |
| `labels` | `default_vocabulary` — 신규 데이터셋의 초기 라벨 사전 (subtask 미정의 시 fallback) |
| `subtasks` / `objects` / `targets` | subtask 프롬프트 템플릿과 슬롯 사전 (아래 참고) |
| `edits` | `history_depth` — `.edits_history/` 보관 개수 |

`ros__parameters:` 루트가 없는 **평범한 YAML** 입니다 (ROS 파라미터 파일이 아님).
실수로 ROS 파라미터 파일을 지정하면 시작 시 알려줍니다.

`--root` / `--host` / `--port` 로 YAML 값을 덮어쓸 수 있고,
실행 중에는 좌측 상단 **Dataset Root** 입력란 + `폴더 열기`(Enter)로
다른 데이터셋 디렉터리로 전환할 수 있습니다.

의존성: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `h5py`, `ffmpeg`(+`ffprobe`)

## 영상 재생 — H.264 프록시

레코더는 `hevc_nvenc`(H.265)로 저장하는데, 리눅스 브라우저는 보통 `<video>` 에서
HEVC 를 디코딩하지 못합니다. 그래서 mp4 를 처음 요청할 때 H.264 로 변환해
데이터셋 폴더의 `.proxy_cache/` 에 캐시합니다.

> **주의**: 시스템에 HEVC 코덱(ffmpeg, libde265 등)을 설치해도 브라우저 재생에는
> 영향이 없습니다. Chrome/Firefox 는 시스템 코덱이 아니라 자체 내장 디코더를
> 쓰기 때문입니다. 브라우저가 HEVC 를 재생한다면 `video.proxy: never` 로 두어
> 변환 없이 원본을 그대로 받을 수 있습니다.

- 원본 mp4 는 건드리지 않습니다 (아카이브 사본).
- 프레임 수를 그대로 보존하므로 UI 의 프레임 인덱스가 HDF5 와 계속 일치합니다.
- NVENC 사용 시 에피소드당 약 1초, 이후 캐시 히트는 즉시.
- 캐시 키에 원본 mtime/크기가 들어가 재수집 시 자동 무효화됩니다.
- `.proxy_cache/` 는 언제든 삭제해도 됩니다 (재생성됨).
- 캐시 유효성은 파일 크기가 아니라 **패킷 수를 원본과 대조**해 확인합니다.
  `+faststart` 때문에 잘린 파일도 헤더상 duration 은 멀쩡해 보이므로,
  크기나 duration 만으로는 중간에 끊긴 인코딩을 걸러낼 수 없습니다.
  잘린 프록시는 자동으로 다시 인코딩됩니다.
- 인코딩 중 클라이언트가 끊어도 임시 `.part` 파일이 남지 않으며,
  이전 실행에서 남은 것은 시작 시 정리합니다.
- `video.proxy` 로 동작 변경: `auto`(기본, 필요할 때만) / `never`(항상 원본) /
  `always`(항상 변환). `playable_codecs` 에 `hevc` 를 추가하면 HEVC 만 원본으로
  내보내고 나머지는 계속 변환합니다.

## 편집 모델 — 비파괴

원본 `data.hdf5` / `*.mp4` / `tasks.json` 은 **절대 수정되지 않습니다.**
모든 편집은 에피소드 폴더의 `edits.json` 사이드카에 기록됩니다.

```jsonc
{
  "version": 1,
  "main_prompt": "pick up the black object and place it in the tray",
  "score": 0.9,              // 에피소드 점수 0~1 (미입력 시 1.0)
  "rejected": false,         // true 면 빌드에서 제외
  "notes": "",
  "trim": {"start": 91, "end": 949},   // 유효 프레임 구간 [start, end)
  "segments": [              // 가변 길이 subtask 구간
    {"start": 91, "end": 286, "label": "reach", "prompt": "...", "score": 1.0}
  ],
  "updated_at": "2026-09-16T09:26:00+0900"
}
```

`edits.json` 이 없으면 `tasks.json` 값으로 기본값이 채워집니다.
저장 시마다 직전 5개 버전이 `.edits_history/` 에 보관됩니다.

라벨 사전은 데이터셋 폴더의 `label_vocabulary.json` 에 저장되어,
자유 텍스트로 인한 라벨 분산(`pick` vs `picking`)을 막습니다.

### Subtask 템플릿

`editor_config.yaml` 의 `subtasks` 는 **키 = segment `label`**,
**값 = segment `prompt` 로 전개되는 템플릿** 입니다.
플레이스홀더는 `{object}` 와 `{target}` 두 가지뿐이고, 각각
`objects:` / `targets:` 리스트에서 인스펙터의 드롭다운으로 채워집니다.

```yaml
subtasks:
  pick:  'pick up {object}'
  place: 'place {object} in {target}'
objects: ['black plastic object']
targets: ['green tray', 'white tray']
```

Object `black plastic object`, Target `green tray` 를 고른 상태에서
`place` 를 누르면 선택된 segment 가 이렇게 채워집니다:

```json
{"label": "place", "prompt": "place black plastic object in green tray"}
```

짧은 `label` 은 통계·필터·학습 조건용으로 안정적으로 남고,
전개된 `prompt` 는 VLA 학습에 그대로 쓰입니다. 키 `1`–`9` 로 앞의 9개를
선택된 segment 에 바로 적용할 수 있습니다.

`subtasks` 가 비어 있으면 인스펙터의 자유 텍스트 vocabulary 로 되돌아갑니다.
리스트가 비어 있는 슬롯을 참조하는 템플릿은 시작 시 오류로 잡힙니다.

## 기능

- 3-cam mp4 동기 재생 (HTTP Range 스트리밍, 프레임 단위 시크)
- HDF5 시계열 플롯 — 벡터 데이터를 **성분별로 전부** 표시 (기본 43개 트레이스) + 재생 헤드 동기
- `stale` 비트마스크 프레임을 타임라인에 마커로 표시
- **Auto-segment**: gripper 개폐 전이로 subtask 경계 제안 + 앞/뒤 정지 구간 자동 trim 제안.
  경계는 전이 지점 그대로가 아니라 **닫힘(grasp) 은 `close_lead_sec` 만큼 앞으로,
  열림(release) 은 `open_lag_sec` 만큼 뒤로** 밀어서 자릅니다
  (기본: 닫힘 0.5초, 열림 1.0초).
  grasp 직전 접근과 release 직후 후퇴는 이웃 subtask 에 속하기 때문입니다
- 가변 길이 segment 라벨링 (label / prompt / score)
- subtask 템플릿 한 번 클릭으로 `label` + `prompt` 동시 채우기 (키 `1`–`9`)
- 우측 인스펙터 2분할 — 왼쪽은 에피소드 필드 + subtask, 오른쪽은 segment 목록.
  각각 따로 스크롤되어 segment 가 많아져도 subtask 버튼이 밀려나지 않습니다
  (창 너비 1500px 이하에서는 위아래로 자동 전환)
- 재생헤드가 들어간 segment 자동 선택 (플롯 클릭·스크럽·재생 모두)
- **Trim to segments** (기본 켜짐) — 에피소드 구간을 첫 segment 시작 ~ 마지막 segment 끝으로
  자동 설정. segment 를 추가/삭제/드래그하면 즉시 따라갑니다. 체크를 풀면 Trim start/end 를
  직접 입력할 수 있고, 이 설정은 브라우저에 기억됩니다
- segment 길이 조절: 플롯에서 경계를 **드래그**, 또는 `[` `]` `{` `}` / nudge 버튼으로 1프레임씩.
  `ripple` 을 켜두면 맞닿은 이웃 segment 경계가 함께 움직여 빈틈이 생기지 않습니다
- 에피소드 prompt, score, reject 플래그, 메모
- score 는 에피소드·segment 모두 **기본 1.0**, 조절 단위 **0.2** (`1 → .8 → .6 → .4 → .2 → 0`).
  비워두면 1.0 으로 저장되고, 명시적으로 넣은 `0` 은 그대로 유지됩니다
- 에피소드 선택 시 3-cam 자동 재생 (상단 `자동재생` 체크박스로 끄기, 설정 유지됨)
- 이전 에피소드 라벨 복사 (`⧉ Prev`)
- 필터: `unlabeled`, `rejected`, 또는 임의 문자열

## 플롯 채널

`reduce: components` 는 벡터 데이터를 성분별 트레이스로 펼칩니다.
관절은 `metadata.json` 의 `joint_names` 를 그대로 쓰고(없거나 개수가 다르면
`joint1..N`), 3벡터는 `x/y/z`, 쿼터니언은 `qx/qy/qz/qw`, 6D 트위스트는
`vx/vy/vz/wx/wy/wz`, rot6d 는 `r11..r32` 로 이름이 붙습니다.

기본 설정 기준 **43개 트레이스 / 11개 그룹**:
`gripper cmd·pos·effort·raw`, `joint pos`(7), `joint vel`(7), `eef pos`(3),
`eef quat`(4), `eef rot6d`(6), `eef vel`(6), `cmd vel`(6).

- 같은 그룹의 성분은 **y 축 스케일을 공유**하므로 joint1~7 을 바로 비교할 수 있고,
  밝기로 성분을 구분합니다. 행 라벨에 그룹의 실제 값 범위가 표시됩니다.
- 플롯 아래 **칩**으로 그룹을 켜고 끌 수 있습니다 (`all` / `none` 포함).
- **행 높이는 창 높이에 맞춰 자동 계산**되어 스크롤바 없이 전부 한 화면에
  들어갑니다. 그룹을 숨기면 남은 그룹이 그만큼 높아지므로, 특정 신호를
  자세히 보려면 나머지를 꺼 두세요 (11개 → 6개 → 3개로 줄일수록 행이 커짐).
- 행이 높아지면 라벨 크기와 선 두께도 함께 커집니다.

일부만 보고 싶으면 `editor_config.yaml` 의 `timeline.channels` 에서 줄이거나,
`reduce` 를 `norm`/`norm3` 으로 바꿔 크기 하나로 합칠 수 있습니다.

## 단축키

| 키 | 동작 |
|---|---|
| `J` / `L` | 1 프레임 뒤/앞 |
| `K` / `Space` | 재생 / 일시정지 |
| `←` / `→` | 10 프레임 (Shift: 30) |
| `I` / `O` | 구간 시작 / 끝 지정 → segment 생성 |
| `S` | 재생 헤드 위치에서 segment 분할 |
| `1`–`9` | 선택된 segment 에 라벨 사전 n번째 라벨 적용 |
| `Del` | 선택된 segment 삭제 |
| `Ctrl+S` | 저장 |

## API

| Method | Path |
|---|---|
| GET | `/api/datasets` |
| GET | `/api/subtasks` |
| GET | `/api/datasets/{ds}` |
| GET | `/api/datasets/{ds}/episodes/{ep}` |
| PUT | `/api/datasets/{ds}/episodes/{ep}/edits` |
| GET | `/api/datasets/{ds}/episodes/{ep}/suggest` |
| GET | `/api/datasets/{ds}/episodes/{ep}/video/{file}` (Range 지원) |
| GET | `/api/datasets/{ds}/episodes/{ep}/file/{file}` |
| PUT | `/api/datasets/{ds}/vocabulary` |
| POST | `/api/datasets/{ds}/bulk_prompt` |

## 패키지 내 관련 도구

- `scripts/merge_raw_episode.py` — 원본 데이터셋 물리적 통합(에피소드 재번호).
- `scripts/convert_to_lerobot.py` — LeRobot 3.0 변환. 아래 참고.

## LeRobot 3.0 변환

```bash
python3 scripts/convert_to_lerobot.py --dry-run              # 계획만 출력
python3 scripts/convert_to_lerobot.py --out <경로>           # 변환
python3 scripts/convert_to_lerobot.py --out <경로> --limit 3 # 스모크 테스트
```

설정은 `config/data_convert_lerobot_config.yaml` 이며, `merge_target_dir` 하위
**모든 데이터셋의 모든 에피소드**를 LeRobot 3.0 데이터셋 하나로 합칩니다.

이 편집기와의 연결: 에피소드에 `edits.json` 이 있으면 **`rejected` 는 제외,
`trim` 은 적용, `main_prompt` 는 `tasks.json` 보다 우선**합니다. 즉 웹에서 편집한
결과가 그대로 학습셋에 반영됩니다.

### 영상은 재인코딩하지 않습니다

`add_frame()` 을 쓰면 mp4 를 전부 디코딩 후 재인코딩하게 됩니다. 대신 레코더가
이미 만들어 둔 에피소드별 mp4 를 LeRobot 에 그대로 넘겨 **stream-copy 로
이어붙입니다.** 원본 HEVC 화질 그대로이고 픽셀이 비트 단위로 동일합니다.

### 알려진 제약

`timestamp` 는 `i/fps` 로 새로 씁니다. 수집된 시각은 30Hz 격자에서 드리프트해
(median 0.6ms, 최대 18.5ms) **292개 에피소드 전부가 LeRobot 의 tolerance(1e-4s)를
초과**하는데, LeRobot 은 `timestamp` 로 영상 프레임을 찾기 때문에 그대로 쓰면
프레임 조회가 깨집니다. 원본 시각은 `observation.timestamp_raw` 로 보존됩니다.

`subtask_score` 의 NaN(미라벨링)은 통계 오염을 막기 위해 -1.0 으로 바뀝니다.

### 학습 시 디코더

이 환경의 torchcodec 은 NVIDIA PyTorch 빌드와 ABI 가 맞지 않아 로드되지 않습니다
(`undefined symbol: _ZN3c1013MessageLogger...`). pyav 로 폴백되며 HEVC 재생에
문제 없습니다. `LEROBOT_VIDEO_BACKEND=pyav` 로 강제할 수도 있습니다.
(`thirdparty/lerobot` 에 pyav 15 호환 패치가 적용되어 있습니다.)
