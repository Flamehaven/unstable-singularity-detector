#!/bin/bash
# Run script for Unstable Singularity Detector Docker container
# Usage: ./run.sh [command] [args...]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

IMAGE="unstable-singularity-detector:latest"

echo -e "${GREEN}[*] Running Unstable Singularity Detector${NC}"
echo ""

# Check if image exists
if ! docker image inspect $IMAGE &> /dev/null; then
    echo -e "${YELLOW}[!] Image not found. Building first...${NC}"
    ./build.sh
fi

# Default: interactive shell
if [ $# -eq 0 ]; then
    echo -e "${GREEN}[*] Starting interactive shell${NC}"
    docker run -it --rm \
        --gpus all \
        -v "$(pwd)/outputs:/app/outputs" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/checkpoints:/app/checkpoints" \
        $IMAGE \
        /bin/bash
else
    # Run with provided command
    echo -e "${GREEN}[*] Running: $@${NC}"
    docker run -it --rm \
        --gpus all \
        -v "$(pwd)/outputs:/app/outputs" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/checkpoints:/app/checkpoints" \
        $IMAGE \
        "$@"
fi