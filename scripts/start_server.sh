#!/bin/bash
# Start ALL servers needed for LIBERO-PRO evaluation:
#   - Perception + LLM proxy (in .venv):   SAM3 (8114), GraspNet (8115),
#     PyRoKi (8116), OpenRouter LLM proxy (8110)
#   - MolMo pointing model via vLLM (in .venv-libero): port 8122
#
# Assumes NO virtual environment is active; activates each as needed.
#
# Usage: bash scripts/start_server.sh [options]
#   --molmo-model NAME   MolMo model served by vLLM (default: allenai/Molmo2-8B)
#   --molmo-port N       MolMo vLLM port (default: 8122)
#   --molmo-gpu-mem F    vLLM --gpu-memory-utilization (default: 0.4, single-GPU safe)
#   --llm-port N         OpenRouter LLM proxy port (default: 8110)
#   --no-molmo           Skip MolMo (e.g. for privileged evals that don't need it)
set -e

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

MOLMO_MODEL="allenai/Molmo2-8B"
MOLMO_PORT=8122
MOLMO_GPU_MEM=0.4
LLM_PORT=8110
START_MOLMO=true

while [ $# -gt 0 ]; do
    case "$1" in
        --molmo-model) MOLMO_MODEL="$2"; shift 2 ;;
        --molmo-port)  MOLMO_PORT="$2";  shift 2 ;;
        --molmo-gpu-mem) MOLMO_GPU_MEM="$2"; shift 2 ;;
        --llm-port)    LLM_PORT="$2";    shift 2 ;;
        --no-molmo)    START_MOLMO=false; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Return the HTTP code for a port's endpoint, or "000" if unreachable.
http_code() {
    curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 "$1" 2>/dev/null || echo "000"
}
# True if the given URL responds at all.
is_up() {
    curl -sf -o /dev/null --connect-timeout 2 "$1" 2>/dev/null
}

echo "=== Starting perception + LLM proxy servers (.venv) ==="
source .venv/bin/activate

# SAM3 server (GPU)
if ! is_up "http://127.0.0.1:8114/"; then
    echo "Starting SAM3 on port 8114..."
    nohup python -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1 > logs/sam3.log 2>&1 &
    echo "  SAM3 PID: $!"
else
    echo "SAM3 (8114): already UP"
fi

# GraspNet server (GPU)
if ! is_up "http://127.0.0.1:8115/"; then
    echo "Starting GraspNet on port 8115..."
    nohup python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1 > logs/graspnet.log 2>&1 &
    echo "  GraspNet PID: $!"
else
    echo "GraspNet (8115): already UP"
fi

# PyRoKi server (CPU)
if ! is_up "http://127.0.0.1:8116/"; then
    echo "Starting PyRoKi on port 8116..."
    nohup python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand > logs/pyroki.log 2>&1 &
    echo "  PyRoKi PID: $!"
else
    echo "PyRoKi (8116): already UP"
fi

# OpenRouter LLM proxy
if ! is_up "http://127.0.0.1:$LLM_PORT/health"; then
    if [ ! -f .openrouterkey ]; then
        echo "WARNING: .openrouterkey not found - LLM proxy will fail to start." >&2
        echo "         Create it with: echo 'sk-or-v1-...' > .openrouterkey" >&2
    fi
    echo "Starting OpenRouter LLM proxy on port $LLM_PORT..."
    nohup python -m capx.serving.openrouter_server --key-file .openrouterkey --port "$LLM_PORT" > logs/llm_proxy.log 2>&1 &
    echo "  LLM proxy PID: $!"
else
    echo "LLM proxy ($LLM_PORT): already UP"
fi

# MolMo via vLLM (.venv-libero)
if [ "$START_MOLMO" = true ]; then
    echo ""
    echo "=== Starting MolMo vLLM server (.venv-libero) ==="
    source .venv-libero/bin/activate
    if ! is_up "http://127.0.0.1:$MOLMO_PORT/v1/models"; then
        echo "Starting MolMo ($MOLMO_MODEL) on port $MOLMO_PORT (gpu-mem $MOLMO_GPU_MEM)..."
        # vLLM serve directly: Molmo needs --trust-remote-code (custom modeling
        # code) and a large multimodal token budget. The capx.serving.vllm_server
        # wrapper does not expose these flags, so call the CLI on PATH.
        nohup vllm serve "$MOLMO_MODEL" \
            --trust-remote-code \
            --port "$MOLMO_PORT" \
            --dtype bfloat16 \
            --max-num-batched-tokens 36864 \
            --limit-mm-per-prompt.image 2 \
            --gpu-memory-utilization "$MOLMO_GPU_MEM" > logs/molmo.log 2>&1 &
        echo "  MolMo PID: $!"
    else
        echo "MolMo ($MOLMO_PORT): already UP"
    fi
fi

echo ""
echo "Waiting 30s for servers to come up (MolMo/vLLM may take longer)..."
sleep 30

echo ""
echo "=== Server health summary ==="
for spec in "8114:SAM3:/" "8115:GraspNet:/" "8116:PyRoKi:/" "$LLM_PORT:LLM-proxy:/health"; do
    port="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; path="${rest#*:}"
    code=$(http_code "http://127.0.0.1:$port$path")
    echo "  $name ($port): $([ "$code" != "000" ] && echo "UP (HTTP $code)" || echo "DOWN")"
done
if [ "$START_MOLMO" = true ]; then
    code=$(http_code "http://127.0.0.1:$MOLMO_PORT/v1/models")
    echo "  MolMo ($MOLMO_PORT): $([ "$code" != "000" ] && echo "UP (HTTP $code)" || echo "DOWN - still loading? tail -f logs/molmo.log")"
fi

echo ""
echo "Logs: logs/{sam3,graspnet,pyroki,llm_proxy,molmo}.log"
echo "Run the evaluation with: bash scripts/evaluate_libero.sh"
