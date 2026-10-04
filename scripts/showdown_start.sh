#!/usr/bin/env bash
# Start the local Pokémon Showdown server for agent training/eval.
# Loopback only, security/rate limits disabled -- for local bot battles, nothing public.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="$ROOT/showdown"

[ -d "$SHOWDOWN_DIR" ] || { echo "showdown/ missing -- run scripts/showdown_setup.sh first" >&2; exit 1; }

cd "$SHOWDOWN_DIR"
exec node pokemon-showdown start --no-security "$@"
