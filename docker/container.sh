#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"

COMMAND="${1:-help}"

CONTAINER_NAME="max"
IMAGE_NAME="moonjongsul/max:pytorch-25.12-x86"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

show_help() {
    cat <<EOF
Usage:
  ${SCRIPT_NAME} <command>

Commands:
  help                  Show this help message
  create                Create and start container
  start                 Start existing container
  enter                 Enter running container
  stop                  Stop container
  remove                Stop and remove container
  purge                 Stop and remove container, networks, volumes
  restart               Restart container
  status                Show container status
  logs                  Show recent logs
  pull                  Pull latest image defined in compose file

Internal config:
  container_name        ${CONTAINER_NAME}
  image_name            ${IMAGE_NAME}

Examples:
  ${SCRIPT_NAME} create
  ${SCRIPT_NAME} enter
  ${SCRIPT_NAME} stop
EOF
}

log() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker command not found"
    docker info >/dev/null 2>&1 || die "docker daemon is not accessible. Check Docker service or permissions"
    docker compose version >/dev/null 2>&1 || die "docker compose is not available"
}

require_compose_file() {
    [[ -f "${COMPOSE_FILE}" ]] || die "Compose file not found: ${COMPOSE_FILE}"
}

compose_run() {
    CONTAINER_NAME="${CONTAINER_NAME}" \
    IMAGE_NAME="${IMAGE_NAME}" \
    M_AX_ROOT="$(dirname "${SCRIPT_DIR}")" \
    docker compose -f "${COMPOSE_FILE}" "$@"
}

is_container_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

is_container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

setup_x11() {
    if [[ -n "${DISPLAY:-}" ]]; then
        log "Setting up X11 forwarding (xhost)..."
        xhost +local:docker >/dev/null 2>&1 || warn "xhost failed. X11 may not work."
    else
        warn "DISPLAY is not set. X11 forwarding will not be available."
    fi
}

enter_shell() {
    docker exec -it "${CONTAINER_NAME}" bash 2>/dev/null || docker exec -it "${CONTAINER_NAME}" sh
}

pull_image() {
    log "Pulling image '${IMAGE_NAME}'..."
    compose_run pull
    log "Pull completed."
}

create_container() {
    if is_container_exists; then
        die "Container '${CONTAINER_NAME}' already exists. Use '${SCRIPT_NAME} start' or '${SCRIPT_NAME} remove' first."
    fi

    setup_x11
    log "Creating and starting container '${CONTAINER_NAME}' with image '${IMAGE_NAME}'..."
    compose_run up -d
    log "Container created and started."
    enter_shell
}

start_container() {
    if ! is_container_exists; then
        die "Container '${CONTAINER_NAME}' does not exist. Run '${SCRIPT_NAME} create' first."
    fi

    if is_container_running; then
        log "Container '${CONTAINER_NAME}' is already running."
        return 0
    fi

    setup_x11
    log "Starting container '${CONTAINER_NAME}'..."
    compose_run start
    log "Container started."
    enter_shell
}

enter_container() {
    if ! is_container_exists; then
        die "Container '${CONTAINER_NAME}' does not exist. Run '${SCRIPT_NAME} create' first."
    fi

    if ! is_container_running; then
        die "Container '${CONTAINER_NAME}' is not running. Run '${SCRIPT_NAME} start' first."
    fi

    setup_x11
    log "Entering container '${CONTAINER_NAME}'..."
    enter_shell
}

stop_container() {
    if ! is_container_exists; then
        log "Container '${CONTAINER_NAME}' does not exist."
        return 0
    fi

    if ! is_container_running; then
        log "Container '${CONTAINER_NAME}' is already stopped."
        return 0
    fi

    log "Stopping container '${CONTAINER_NAME}'..."
    compose_run stop
    log "Container stopped."
}

remove_container() {
    if ! is_container_exists; then
        log "Container '${CONTAINER_NAME}' does not exist. Nothing to remove."
        return 0
    fi

    warn "This will stop and remove container '${CONTAINER_NAME}'."
    warn "Named volumes are preserved."
    read -r -p "Type 'yes' to continue: " answer

    if [[ "${answer}" == "yes" ]]; then
        compose_run down
        log "Container removed."
    else
        log "Operation cancelled."
    fi
}

purge_container() {
    warn "This will stop and remove container '${CONTAINER_NAME}', related networks, and named volumes."
    warn "This may permanently delete stored data."
    read -r -p "Type 'purge' to continue: " answer

    if [[ "${answer}" == "purge" ]]; then
        compose_run down -v --remove-orphans
        log "Container, media, and volumes removed."
    else
        log "Operation cancelled."
    fi
}

restart_container() {
    if ! is_container_exists; then
        die "Container '${CONTAINER_NAME}' does not exist."
    fi

    setup_x11
    log "Restarting container '${CONTAINER_NAME}'..."
    compose_run restart
    log "Container restarted."
}

status_container() {
    if ! is_container_exists; then
        log "Container '${CONTAINER_NAME}' does not exist."
        return 0
    fi

    if is_container_running; then
        log "Container '${CONTAINER_NAME}' is running."
    else
        log "Container '${CONTAINER_NAME}' exists but is stopped."
    fi

    docker ps -a --filter "name=^${CONTAINER_NAME}$"
}

logs_container() {
    if ! is_container_exists; then
        die "Container '${CONTAINER_NAME}' does not exist."
    fi

    compose_run logs --tail=200 -f
}

main() {
    require_docker
    require_compose_file

    case "${COMMAND}" in
        help)    show_help ;;
        create)  create_container ;;
        start)   start_container ;;
        enter)   enter_container ;;
        stop)    stop_container ;;
        remove)  remove_container ;;
        purge)   purge_container ;;
        restart) restart_container ;;
        status)  status_container ;;
        logs)    logs_container ;;
        pull)    pull_image ;;
        *)
            echo "[ERROR] Unknown command: '${COMMAND}'" >&2
            echo
            show_help
            exit 1
            ;;
    esac
}

main "$@"
