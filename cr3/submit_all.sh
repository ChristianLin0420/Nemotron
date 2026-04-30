#!/bin/bash
# Submit the CR3-on-Nemotron sweep, optionally one wave at a time.
#
# A "wave" is one of the 4 CR3 datasets: assy17 / c310 / tp00302 / tp00303.
# Each wave is 7 sbatch jobs. Submitting in waves lets you gate the next
# wave on the previous wave's accuracy without burning compute up front.
#
# Usage:
#   bash Nemotron/cr3/submit_all.sh --wave assy17       # only ASSY17 (7 jobs)
#   bash Nemotron/cr3/submit_all.sh --wave c310
#   bash Nemotron/cr3/submit_all.sh --wave tp00302
#   bash Nemotron/cr3/submit_all.sh --wave tp00303
#   bash Nemotron/cr3/submit_all.sh --all               # all 28 jobs at once
#   bash Nemotron/cr3/submit_all.sh --list              # dry-run, just print
#
# Sleeps 5 minutes between submissions in a wave so the cluster scheduler
# doesn't see a burst of identical jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DIR="$SCRIPT_DIR/sbatch"

WAVE="${1:-}"
shift || true

ALL_DATASETS=(assy17 c310 tp00302 tp00303)

case "$WAVE" in
    --wave)
        WAVE_NAME="${1:-}"
        shift
        case "$WAVE_NAME" in
            assy17|c310|tp00302|tp00303) DATASETS=("$WAVE_NAME");;
            *) echo "Unknown wave: $WAVE_NAME" >&2; exit 2;;
        esac
        ;;
    --all)
        DATASETS=("${ALL_DATASETS[@]}")
        ;;
    --list)
        for ds in "${ALL_DATASETS[@]}"; do
            echo "[$ds]"
            ls "$SBATCH_DIR/$ds/"sbatch_*.sh 2>/dev/null | sed 's/^/  /'
        done
        exit 0
        ;;
    --help|-h|"")
        sed -n '1,/^# Sleeps/p' "$0"
        exit 0
        ;;
    *)
        echo "Unknown option: $WAVE" >&2
        exit 2
        ;;
esac

SUBMIT_LIST=()
for ds in "${DATASETS[@]}"; do
    if [[ ! -d "$SBATCH_DIR/$ds" ]]; then
        echo "WARN: $SBATCH_DIR/$ds does not exist (run gen_cr3_configs.py first)" >&2
        continue
    fi
    for sb in "$SBATCH_DIR/$ds/"sbatch_*.sh; do
        [[ -f "$sb" ]] && SUBMIT_LIST+=("$sb")
    done
done

if (( ${#SUBMIT_LIST[@]} == 0 )); then
    echo "No sbatch files found. Did you run gen_cr3_configs.py?" >&2
    exit 1
fi

echo "Submitting ${#SUBMIT_LIST[@]} jobs:"
for sb in "${SUBMIT_LIST[@]}"; do
    echo "  $sb"
done
echo

for i in "${!SUBMIT_LIST[@]}"; do
    sb="${SUBMIT_LIST[$i]}"
    echo "[$((i+1))/${#SUBMIT_LIST[@]}] sbatch $sb"
    sbatch "$sb"
    if (( i < ${#SUBMIT_LIST[@]} - 1 )); then
        echo "  sleeping 5 min before next submission ..."
        sleep 300
    fi
done

echo
echo "Done. Tail logs with: tail -f logs/cr3-nemotron-*/stdout.log"
