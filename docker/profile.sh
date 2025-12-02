#!/bin/bash

# Helper script to run Scalene profiling inside the docker container
# Usage: ./docker/profile.sh path/to/script.py

if [ -z "$1" ]; then
    echo "Usage: $0 path/to/script.py"
    exit 1
fi

SCRIPT_PATH=$1
FILENAME=$(basename "$SCRIPT_PATH")
BASENAME="${FILENAME%.*}"

# Ensure we are running from project root
if [ ! -f "docker/docker-compose.yml" ]; then
    echo "Error: Please run this script from the project root directory."
    echo "Example: ./docker/profile.sh docker/demo_observability.py"
    exit 1
fi

echo "🚀 Starting profiling for $SCRIPT_PATH..."
echo "----------------------------------------"

# Run Scalene inside the app container
# --html: Generate HTML output
# --outfile: Save to current directory (mounted volume)
docker compose -f docker/docker-compose.yml run --rm app \
    scalene --html --outfile "profile_${BASENAME}.html" "$SCRIPT_PATH"

echo "----------------------------------------"
echo "✅ Profiling complete!"
echo "📄 View report: profile_${BASENAME}.html"

