#!/usr/bin/env bash
set -euo pipefail

uv run vlm-infer \
  --image "${IMAGE_PATH:?set IMAGE_PATH}" \
  --prompt-file "${PROMPT_FILE:?set PROMPT_FILE}"
