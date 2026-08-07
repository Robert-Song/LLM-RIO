#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="0.12.1"
DIAGNOSTIC_DIR="${PROJECT_DIR}/diagnostics"
DIAGNOSTIC_FILE="${DIAGNOSTIC_DIR}/setup-$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${DIAGNOSTIC_DIR}"
exec > >(tee -a "${DIAGNOSTIC_FILE}") 2>&1

failed_stage="preflight"
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "SETUP_FAILED stage=%s status=%s log=%s\n" "${failed_stage}" "${status}" "${DIAGNOSTIC_FILE}"; fi' EXIT

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "LLM-RIO production setup requires Linux."
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Ask HPC administration to install/repair the NVIDIA driver."
  exit 3
fi

failed_stage="uv_install"
if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "uv and curl are unavailable. Install uv in user space, then rerun setup.sh."
    exit 4
  fi
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

failed_stage="environment"
cd "${PROJECT_DIR}"
uv venv --python 3.12
uv sync --extra engine

failed_stage="configuration"
if [[ ! -f config.toml ]]; then
  cp config.example.toml config.toml
fi
mkdir -p state models logs
for path in state models logs; do
  if [[ ! -w "${path}" ]]; then
    echo "Path is not writable: ${PROJECT_DIR}/${path}"
    exit 5
  fi
done

failed_stage="inventory"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
uv run llm-rio doctor --json

failed_stage="engine_import"
uv run python -c 'import vllm; print("vllm", vllm.__version__)'

cat <<EOF
LLM-RIO setup completed.
Diagnostic report: ${DIAGNOSTIC_FILE}
Review config.toml, then run ./llmctl serve.
The first startup creates and visibly prints the initial administrator key.
The endpoint-level inference smoke test is intentionally deferred until a model and test API key are supplied.
EOF

