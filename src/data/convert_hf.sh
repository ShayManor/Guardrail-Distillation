#!/bin/bash

set -e

HF_USERNAME="shaymanor"

ZIPS=(
    "leftImg8bit_trainvaltest.zip"
    "gtFine_trainvaltest.zip"
    "gtCoarse.zip"
)

DOWNLOADS=~/Downloads

for zip in "${ZIPS[@]}"; do
    name="${zip%.zip}"
    echo "==> Unzipping $zip..."
    unzip -q "$DOWNLOADS/$zip" -d "$DOWNLOADS/$name"

    echo "==> Uploading $name to huggingface.co/$HF_USERNAME/$name..."
    huggingface-cli upload "$HF_USERNAME/$name" "$DOWNLOADS/$name" \
        --repo-type dataset \
        --private

    echo "==> Done: $name"
done

echo "All uploads complete."