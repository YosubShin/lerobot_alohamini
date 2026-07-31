#!/usr/bin/env bash
# Rolling mover: during long collection sessions, pull completed fisheye
# archive segments off the Pi (the SD card holds <1 h of MJPG at ~19 GB/hour).
# The host writes 5-min segments; the newest .mkv is still being written, so
# everything older is safe to move. Run this on the workstation and leave it.
#
#   ./fisheye_archive_mover.sh [user@pi] [dest_dir]
set -euo pipefail
PI=${1:-yosub@192.168.0.50}
DEST=${2:-/mnt/nvme/lerobot/fisheye_archive}
mkdir -p "$DEST"
echo "Moving completed segments from $PI:~/fisheye_archive to $DEST (Ctrl-C to stop)"
while true; do
  files=$(ssh "$PI" 'ls -t ~/fisheye_archive/*.mkv 2>/dev/null | tail -n +2') || files=""
  if [ -n "$files" ]; then
    srcs=()
    for f in $files; do srcs+=("$PI:$f"); done
    rsync -a --remove-source-files "${srcs[@]}" "$DEST/" \
      && echo "$(date +%H:%M:%S) moved: $(echo "$files" | xargs -n1 basename | tr '\n' ' ')"
  fi
  sleep 60
done
