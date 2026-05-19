#!/usr/bin/env bash

set -euo pipefail

REMOTE_ALIAS="runpod"
REMOTE_ROOT="/workspace/gp-shield"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

push_path() {
  local relative_path="$1"
  local normalized_path="${relative_path%/}"
  local source_path
  local remote_parent

  if [[ -z "${normalized_path}" ]]; then
    echo "Path must not be empty: ${relative_path}" >&2
    exit 1
  fi

  source_path="${PROJECT_ROOT}/${normalized_path}"

  if [[ ! -e "${source_path}" ]]; then
    echo "Path does not exist: ${normalized_path}" >&2
    exit 1
  fi

  if [[ "${normalized_path}" = /* ]]; then
    echo "Path must be relative to the project root: ${normalized_path}" >&2
    exit 1
  fi

  remote_parent="$(dirname "${normalized_path}")"
  if [[ "${remote_parent}" = "." ]]; then
    remote_parent=""
  fi

  ssh "${REMOTE_ALIAS}" "mkdir -p '${REMOTE_ROOT}/${remote_parent}'"
  rsync -rvz "${source_path}" "${REMOTE_ALIAS}:${REMOTE_ROOT}/${remote_parent}/"
}

pull_path() {
  local relative_path="$1"
  local normalized_path="${relative_path%/}"
  local local_parent

  if [[ -z "${normalized_path}" ]]; then
    echo "Path must not be empty: ${relative_path}" >&2
    exit 1
  fi

  if [[ "${normalized_path}" = /* ]]; then
    echo "Path must be relative to the project root: ${normalized_path}" >&2
    exit 1
  fi

  local_parent="${PROJECT_ROOT}/$(dirname "${normalized_path}")"
  if [[ "$(dirname "${normalized_path}")" = "." ]]; then
    local_parent="${PROJECT_ROOT}"
  fi

  mkdir -p "${local_parent}"
  rsync -rvz "${REMOTE_ALIAS}:${REMOTE_ROOT}/${normalized_path}" "${local_parent}/"
}

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 [--pull] <path> [<path> ...]" >&2
  echo "  (default: push local → remote)" >&2
  echo "  --pull: pull remote → local" >&2
  exit 1
fi

MODE="push"
if [[ "$1" == "--pull" ]]; then
  MODE="pull"
  shift
  if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 --pull <path> [<path> ...]" >&2
    exit 1
  fi
fi

for relative_path in "$@"; do
  if [[ "${MODE}" == "pull" ]]; then
    pull_path "${relative_path}"
  else
    push_path "${relative_path}"
  fi
done
