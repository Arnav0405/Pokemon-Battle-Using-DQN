#!/usr/bin/env bash
# One-shot setup for the local Pokémon Showdown battle server.
# Idempotent: safe to re-run. Clones the server (gitignored) + writes a loopback config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="$ROOT/showdown"
REPO_URL="https://github.com/smogon/pokemon-showdown.git"

if [ ! -d "$SHOWDOWN_DIR/.git" ]; then
  echo "cloning pokemon-showdown -> $SHOWDOWN_DIR"
  git clone --depth 1 "$REPO_URL" "$SHOWDOWN_DIR"
fi

cd "$SHOWDOWN_DIR"
echo "installing node deps"
npm install --no-audit --no-fund

if [ ! -f config/config.js ]; then
  echo "writing config/config.js from config-example.js"
  cp config/config-example.js config/config.js
fi
node "$ROOT/scripts/patch_showdown_config.js" config/config.js

echo "showdown version: $(node -p "require('./package.json').version")"
echo "done. start the server with: scripts/showdown_start.sh"
