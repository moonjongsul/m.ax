#!/bin/bash

# 1. 이미지 이름 및 태그 설정
SYS_TARGET="x86" # 시스템 타겟 설정 (ex: 'jetpack', 'x86', 'dgx')
BUILD_IMAGE_NAME="moonjongsul/max"
NGC_VERSION="25.12" # NGC PyTorch 버전 (ex: '25.12', '22.03' 등)
DOCKERFILE="Dockerfile.${SYS_TARGET}"
NGC_IMAGE="pytorch:${NGC_VERSION}-py3"
TAG="pytorch-${NGC_VERSION}-${SYS_TARGET}"

# m.ax 프로젝트 루트 (docker/ 의 상위 디렉토리)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "=========================================="
echo "🚀 Docker Build 시작: ${BUILD_IMAGE_NAME}:${TAG}"
echo "파일 경로: ${SCRIPT_DIR}/${DOCKERFILE}"
echo "빌드 컨텍스트: ${PROJECT_ROOT}"
echo "=========================================="

# 2. 빌드 시간 측정 및 실행
# BuildKit 활성화 (병렬 빌드 및 성능 향상)
# --no-cache 옵션이 필요하면 아래 명령어 뒤에 추가하세요.
START_TIME=$(date +%s)

DOCKER_BUILDKIT=1 docker build \
  -f "${SCRIPT_DIR}/${DOCKERFILE}" \
  -t "${BUILD_IMAGE_NAME}:${TAG}" \
  --build-arg CACHE_BUST=$(date +%s) \
  --build-arg NGC_IMAGE=${NGC_IMAGE} \
  --build-arg USER_UID=$(id -u $USER) \
  --build-arg USER_GID=$(id -g $USER) \
  --build-arg USERNAME=$USER \
  "${PROJECT_ROOT}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 3. 결과 출력
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ 빌드 성공!"
    echo "소요 시간: $(($DURATION / 60))분 $(($DURATION % 60))초"
    echo "이미지 이름: ${BUILD_IMAGE_NAME}:${TAG}"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ 빌드 실패! 로그를 확인해 주세요."
    echo "=========================================="
    exit 1
fi
