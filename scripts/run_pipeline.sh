#!/usr/bin/env bash
# Full pipeline for one capture:  ./scripts/run_pipeline.sh my_video.mp4 workspaces/my_scene
set -euo pipefail
SRC=${1:?usage: run_pipeline.sh <video|photo-folder|webcam-index> <workspace> [extra train args]}
WS=${2:?usage: run_pipeline.sh <video|photo-folder|webcam-index> <workspace> [extra train args]}
shift 2

image-to-3d capture "$SRC" "$WS"
image-to-3d sfm "$WS"
image-to-3d init "$WS"
image-to-3d train "$WS" "$@"
image-to-3d render "$WS" --views orbit --video
echo "trained scene: $WS/output/point_cloud.ply   turntable: $WS/renders/orbit.mp4"
