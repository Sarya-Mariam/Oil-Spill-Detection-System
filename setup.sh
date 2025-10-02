#!/bin/bash
mkdir -p models
pip install gdown
# Download model zip from Google Drive (replace ID if needed)
gdown --id 1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg -O models/dual_head_best.zip
# Unzip model into models/ directory
unzip -o models/dual_head_best.zip -d models
