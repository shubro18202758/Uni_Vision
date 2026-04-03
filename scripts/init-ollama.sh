#!/bin/sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uni_Vision — Ollama Model Initializer
#
# Pulls the Gemma 4 E2B Q4_K_M base model, then creates
# three custom Modelfile variants:
#   1. uni-vision-ocr         (primary OCR engine)
#   2. uni-vision-adjudicator (spatial reasoning adjudicator)
#   3. uni-vision-navarasa    (Indian contextualizer — Navarasa 2.0 7B)
#
# The Navarasa model pulls its weights directly from HuggingFace GGUF
# (no separate base-model pull required).
#
# Usage:
#   ./scripts/init-ollama.sh                     # local Ollama
#   OLLAMA_HOST=http://ollama:11434 ./scripts/init-ollama.sh  # Docker
#
# OCR and Adjudicator share the same Q4_K_M Gemma 4 weights (hard-linked
# by Ollama).  Navarasa uses a separate Gemma-7B GGUF from HuggingFace.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODELFILE_DIR="${MODELFILE_DIR:-/config/ollama}"
MAX_RETRIES=30
RETRY_INTERVAL=5

log() {
    printf "[init-ollama] %s\n" "$1"
}

# ── Wait for Ollama to be ready ───────────────────────────────────
wait_for_ollama() {
    log "Waiting for Ollama at ${OLLAMA_HOST} ..."
    retries=0
    while [ "$retries" -lt "$MAX_RETRIES" ]; do
        if curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            log "Ollama is ready."
            return 0
        fi
        retries=$((retries + 1))
        log "  retry ${retries}/${MAX_RETRIES} ..."
        sleep "$RETRY_INTERVAL"
    done
    log "ERROR: Ollama did not become ready within $((MAX_RETRIES * RETRY_INTERVAL))s."
    exit 1
}

# ── Pull base model ──────────────────────────────────────────────
pull_base_model() {
    local model="gemma4:e2b"
    log "Pulling base model: ${model} ..."
    curl -sf "${OLLAMA_HOST}/api/pull" \
        -d "{\"name\": \"${model}\", \"stream\": false}" \
        -H "Content-Type: application/json" \
        -o /dev/null || {
        log "ERROR: Failed to pull ${model}."
        exit 1
    }
    log "Base model ${model} pulled successfully."
}

# ── Create custom model from Modelfile ────────────────────────────
create_model() {
    local name="$1"
    local modelfile_path="$2"

    if [ ! -f "${modelfile_path}" ]; then
        log "ERROR: Modelfile not found at ${modelfile_path}"
        exit 1
    fi

    log "Creating model: ${name} from ${modelfile_path} ..."
    local modelfile_content
    modelfile_content=$(cat "${modelfile_path}")

    curl -sf "${OLLAMA_HOST}/api/create" \
        -d "{\"name\": \"${name}\", \"modelfile\": $(printf '%s' "${modelfile_content}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '%s' "${modelfile_content}" | jq -Rs . 2>/dev/null), \"stream\": false}" \
        -H "Content-Type: application/json" \
        -o /dev/null || {
        log "ERROR: Failed to create model ${name}."
        exit 1
    }
    log "Model ${name} created successfully."
}

# ── Validate model is loadable ────────────────────────────────────
validate_model() {
    local name="$1"
    log "Validating model: ${name} ..."

    local response
    response=$(curl -sf "${OLLAMA_HOST}/api/generate" \
        -d "{\"model\": \"${name}\", \"prompt\": \"test\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
        -H "Content-Type: application/json" 2>&1) || {
        log "WARNING: Validation request for ${name} failed (model may still be loading)."
        return 0
    }
    log "Model ${name} is valid and loadable."
}

# ── List all models ───────────────────────────────────────────────
list_models() {
    log "Installed models:"
    curl -sf "${OLLAMA_HOST}/api/tags" | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    size_gb = m.get('size', 0) / (1024**3)
    print(f\"  - {m['name']}  ({size_gb:.1f} GB)\")
" 2>/dev/null || curl -sf "${OLLAMA_HOST}/api/tags"
}

# ── Main ──────────────────────────────────────────────────────────
main() {
    log "══════════════════════════════════════════════════════"
    log "  Uni_Vision — Ollama Model Initialization"
    log "══════════════════════════════════════════════════════"

    wait_for_ollama

    # Step 1: Pull the Gemma 4 E2B Q4_K_M base
    pull_base_model

    # Step 2: Create OCR variant
    create_model "uni-vision-ocr" "${MODELFILE_DIR}/Modelfile.ocr"

    # Step 3: Create Adjudicator variant
    create_model "uni-vision-adjudicator" "${MODELFILE_DIR}/Modelfile.adjudicator"

    # Step 4: Create Navarasa Indian contextualizer (downloads GGUF from HuggingFace)
    log "Navarasa 2.0 7B will be downloaded from HuggingFace on first creation — this may take a while."
    create_model "uni-vision-navarasa" "${MODELFILE_DIR}/Modelfile.navarasa"

    # Step 5: Validate all three are loadable
    validate_model "uni-vision-ocr"
    validate_model "uni-vision-adjudicator"
    validate_model "uni-vision-navarasa"

    # Step 6: Show installed models
    list_models

    log "══════════════════════════════════════════════════════"
    log "  Initialization complete."
    log "══════════════════════════════════════════════════════"
}

main "$@"
