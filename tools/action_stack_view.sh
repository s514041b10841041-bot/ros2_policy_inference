#!/usr/bin/env bash
# Show the policy's action chunks live in this terminal (see action_stack_view.py). Works from any
# folder, needs no conda activate and no Docker. Extra args go to the viewer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${ROS_ENV:-$HOME/miniconda3/envs/lerobot312-ros}"

# Same DDS, domain and discovery settings as the inference container and launch_isaacsim.sh
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<CycloneDDS><Domain><Discovery><MaxAutoParticipantIndex>100</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
exec "$ENV/bin/python" "$SCRIPT_DIR/action_stack_view.py" "$@"
