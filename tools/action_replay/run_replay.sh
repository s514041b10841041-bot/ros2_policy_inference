#!/usr/bin/env bash
# Runs replay_episode.py in the lerobot_ros2 image (no ROS needed on the host).
# Results (.csv + .html report) go to tools/action_replay/results/, owned by you.
#
# Usage:
#   ./run_replay.sh --episode 12                 # answer the sim's chunk requests with episode 12
#   ./run_replay.sh --episode 12 --gripper open  # any replay_episode.py option (see --help)
#   DATASET=/path/to/other_dataset ./run_replay.sh --episode 0
#
# Uses Fast DDS, the same middleware as Isaac Sim, instead of the image's default
# CycloneDDS: Isaac crashed in libfastrtps at the IL hand-off while a CycloneDDS
# node was subscribed to the IL topics (2026-10-09 15:11). UDPv4 because Fast
# DDS's shared-memory transport doesn't work between the host and a container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELTA="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATASET="${DATASET:-$DELTA/delta_robot_friday/data/Le-007-turbo}"
URDF_DIR="$DELTA/delta_02/delta_extension/assets/motion_policy_configs"
mkdir -p "$SCRIPT_DIR/results"

docker run --rm -it \
    --name episode_replay \
    --network host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e ROS_LOG_DIR=/tmp/ros_log \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "$SCRIPT_DIR":/action_replay \
    -v "$(readlink -f "$DATASET")":/dataset:ro \
    -v "$URDF_DIR":/urdf:ro \
    --entrypoint bash \
    lerobot_ros2:latest -c 'source /opt/ros/jazzy/setup.bash && exec python3 /action_replay/replay_episode.py "$@"' _ "$@"
