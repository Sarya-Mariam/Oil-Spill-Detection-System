#!/bin/bash
set -e

mkdir -p models
pip install gdown

# Download model zip from Google Drive
gdown --id 1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg -O models/model.zip

# Unzip into models/
unzip -o models/model.zip -d models

# If the file exists inside a subfolder, move it up
FOUND_FILE=$(find models -name "dual_head_best.h5" | head -n 1)

if [ -f "$FOUND_FILE" ]; then
  echo "Found model at $FOUND_FILE"
  mv "$FOUND_FILE" models/dual_head_best.h5
else
  echo "❌ dual_head_best.h5 not found inside the zip"
  exit 1
fi
