#!/bin/bash
# Run a LIBERO-PRO evaluation (single suite) via the batch runner.
#
# Assumes the servers are already running (start them with scripts/start_server.sh).
# Activates .venv-libero, which is required for LIBERO + MolMo.
#
# Usage: bash scripts/evaluate_libero.sh [options]
#   --suite NAME       LIBERO-PRO suite (default: libero_object_swap)
#   --trials N         Total trials per task (default: 10)
#   --workers N        Parallel workers (default: 4)
#   --output-dir DIR   Output directory (default: ./outputs/evaluate_libero)
#   --model NAME       Model to query (default: google/gemini-3.1-pro-preview)
#   --config PATH      Base config (default: env_configs/libero/franka_libero_cap_agent0.yaml)
#   --force            Skip the server pre-flight check
set -e

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs outputs

SUITE="libero_object_swap"
TRIALS=10
WORKERS=4
OUTPUT_DIR="./outputs/evaluate_libero"
MODEL="google/gemini-3.1-pro-preview"
CONFIG="env_configs/libero/franka_libero_cap_agent0.yaml"
FORCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --suite)      SUITE="$2";      shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --workers)    WORKERS="$2";    shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model)      MODEL="$2";      shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        --force)      FORCE=true;      shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# True if the given URL responds at all.
is_up() {
    curl -sf -o /dev/null --connect-timeout 2 "$1" 2>/dev/null
}

if [ "$FORCE" != true ]; then
    echo "=== Pre-flight: checking servers ==="
    missing=false
    for spec in "8114:SAM3:/" "8115:GraspNet:/" "8116:PyRoKi:/" "8110:LLM-proxy:/health" "8122:MolMo:/v1/models"; do
        port="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; path="${rest#*:}"
        if is_up "http://127.0.0.1:$port$path"; then
            echo "  $name ($port): UP"
        else
            echo "  $name ($port): DOWN"
            missing=true
        fi
    done
    if [ "$missing" = true ]; then
        echo ""
        echo "Some servers are DOWN. Start them first with:  bash scripts/start_server.sh"
        echo "(Use --force to run anyway, e.g. for privileged configs that don't need MolMo.)"
        exit 1
    fi
fi

echo ""
echo "=== Running LIBERO-PRO evaluation ==="
echo "  Suite:   $SUITE"
echo "  Trials:  $TRIALS   Workers: $WORKERS"
echo "  Model:   $MODEL"
echo "  Config:  $CONFIG"
echo "  Output:  $OUTPUT_DIR"
echo ""

source .venv-libero/bin/activate
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m capx.envs.scripts.run_libero_batch \
    --args.base-config-path "$CONFIG" \
    --args.suites "$SUITE" \
    --args.total-trials "$TRIALS" \
    --args.num-workers "$WORKERS" \
    --args.output-dir "$OUTPUT_DIR" \
    --args.models "$MODEL" \
    --args.record-video False \
    2>&1 | tee "logs/evaluate_libero.log"

echo ""
echo "=== Results ==="
total=$(find "$OUTPUT_DIR" -name 'trial_*' -type d 2>/dev/null | wc -l)
success=$(find "$OUTPUT_DIR" -name '*taskcompleted_1*' -type d 2>/dev/null | wc -l)
echo "  Success: $success / $total trials"
if [ "$total" -gt 0 ]; then
    rate=$(python3 -c "print(f'{$success/$total:.2%}')")
    echo "  Success rate: $rate"
fi
echo "  Artifacts: $OUTPUT_DIR"
