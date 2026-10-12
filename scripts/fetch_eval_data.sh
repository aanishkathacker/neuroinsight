#!/bin/bash
# Fetch the MIT1003 saliency benchmark (Judd et al., ICCV 2009).
# ~235 MB. Not committed to the repo; run this once before `neuroinsight eval`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/eval"
BASE="https://people.csail.mit.edu/tjudd/WherePeopleLook"
mkdir -p "$DEST"
cd "$DEST"
for f in ALLSTIMULI.zip ALLFIXATIONMAPS.zip; do
  [[ -f "$f" ]] && { echo "  have $f"; continue; }
  echo "  downloading $f ..."
  curl -# -L -o "$f" "$BASE/$f"
done
for z in ALLSTIMULI ALLFIXATIONMAPS; do
  [[ -d "$z" ]] || unzip -oq "$z.zip"
done
rm -rf __MACOSX
echo "  ready: $(ls ALLSTIMULI/*.jpeg | wc -l | tr -d ' ') stimuli, $(ls ALLFIXATIONMAPS/*_fixMap.jpg | wc -l | tr -d ' ') fixation maps"
