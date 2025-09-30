#!/bin/bash
# Build script for Unstable Singularity Detector Docker image
# Usage: ./build.sh [--no-cache] [--gpu|--cpu]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[*] Building Unstable Singularity Detector Docker Image${NC}"
echo ""

# Parse arguments
NO_CACHE=""
GPU_SUPPORT="--build-arg CUDA_VERSION=11.8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            echo -e "${YELLOW}[!] Building without cache${NC}"
            shift
            ;;
        --cpu)
            GPU_SUPPORT=""
            echo -e "${YELLOW}[!] Building CPU-only image${NC}"
            shift
            ;;
        --gpu)
            GPU_SUPPORT="--build-arg CUDA_VERSION=11.8"
            echo -e "${GREEN}[*] Building GPU-enabled image${NC}"
            shift
            ;;
        *)
            echo -e "${RED}[-] Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[-] Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Build image
echo -e "${GREEN}[*] Building Docker image...${NC}"
docker build \
    $NO_CACHE \
    $GPU_SUPPORT \
    -t unstable-singularity-detector:1.0.0 \
    -t unstable-singularity-detector:latest \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[+] Build successful!${NC}"
    echo ""
    echo "Image tags:"
    echo "  - unstable-singularity-detector:1.0.0"
    echo "  - unstable-singularity-detector:latest"
    echo ""
    echo "Run with:"
    echo "  docker run -it --rm unstable-singularity-detector:latest"
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up detector"
else
    echo -e "${RED}[-] Build failed!${NC}"
    exit 1
fi