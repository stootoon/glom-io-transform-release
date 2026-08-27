#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 user@host:/path/to/remote/repo"
    exit 1
fi

REMOTE="$1"
LOCAL="$(pwd)"

echo "Syncing PDFs:"
echo "  Local : $LOCAL"
echo "  Remote: $REMOTE"
echo

# Remote -> local
echo "Remote -> local..."
rsync -av --update \
    --include='*/' \
    --include='*.pdf' \
    --exclude='*' \
    "$REMOTE/" "$LOCAL/"

# Local -> remote
echo "Local -> remote..."
rsync -av --update \
    --include='*/' \
    --include='*.pdf' \
    --exclude='*' \
    "$LOCAL/" "$REMOTE/"

echo
echo "Done."
