#!/usr/bin/env bash
# Resilient XTTS v2 download via curl: resume (-C -) + retry-all-errors, wrapped
# in an outer loop so repeated connection resets just keep resuming.
set -u

MD="C:/Users/evanm/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2"
BASE="https://huggingface.co/coqui/XTTS-v2/resolve/main"
FILES="config.json vocab.json hash.md5 speakers_xtts.pth model.pth"
mkdir -p "$MD"

for f in $FILES; do
  echo "=== $f ==="
  ok=0
  for attempt in $(seq 1 40); do
    curl -sL -C - --retry 10 --retry-all-errors --retry-delay 3 \
         --connect-timeout 30 \
         -o "$MD/$f" "$BASE/$f"
    rc=$?
    if [ $rc -eq 0 ]; then ok=1; echo "  ok (attempt $attempt)"; break; fi
    echo "  attempt $attempt failed (curl rc=$rc); resuming..."
    sleep 2
  done
  if [ $ok -ne 1 ]; then echo "FAILED: $f"; exit 1; fi
  echo "  size: $(stat -c %s "$MD/$f" 2>/dev/null) bytes"
done
echo "ALL FILES DOWNLOADED"
