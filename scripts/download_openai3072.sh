#!/usr/bin/env bash
# Current OpenAI-3072 dataset is DBpedia. SimpleWiki has a separate legacy entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/download_dbpedia3072.sh" "$@"
