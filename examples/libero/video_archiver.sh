#!/bin/bash
# 用法: bash examples/libero/video_archiver.sh <video_dir>
# main.py 每集把 rollout 视频覆盖写到固定文件名（rollout_<task>_success|failure.mp4），
# 本脚本轮询目录，把新出现的视频按到达顺序改名归档到 <video_dir>/archived/，
# 从而保留每一集的视频。mv 是 rename 语义，即使 mimwrite 正在写也不丢数据。
# 停止: Ctrl+C 或 kill。
set -u
DIR="$1"
mkdir -p "$DIR/archived"
while true; do
  for f in "$DIR"/rollout_*.mp4; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .mp4)
    n=$(find "$DIR/archived" -name "${base}_ep*.mp4" | wc -l)
    dest="$DIR/archived/${base}_ep$(printf '%03d' "$n").mp4"
    mv "$f" "$dest"
    echo "[archiver] $f -> $dest"
  done
  sleep 2
done
