#!/bin/bash

set -x

ELA_AI_DIR="$(pwd)"
ELA_IMAGE=elaai
ELA_IMAGE_VERSION=0.6
ELA_AI_APP=$1
ELA_ACTIVITY=$2

docker run -it \
    -v "$ELA_AI_DIR:/apps" \
    --workdir /apps \
    --network host \
    --env-file local.env \
    "$ELA_IMAGE":"$ELA_IMAGE_VERSION" \
    "-m${ELA_AI_APP}.app" \
    "$ELA_ACTIVITY"
