#!/usr/bin/env bash
# Create the PRIVATE GitHub repository and push everything (branches + tags).
# Usage: bash scripts/setup_github.sh [repo-name]
set -euo pipefail
NAME="${1:-sce513-spacefilling-sysid}"
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo "Install GitHub CLI first: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login
[ -d .git ] || { git init -b main && git add -A && git commit -m "Initial commit"; }
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main --tags
else
  gh repo create "$NAME" --private --source=. --remote=origin --push \
     --description "SCE 513 (KFUPM) reproduction of Kiss, Toth & Schoukens (SYSID 2024): space-filling input design"
  git push origin --tags
fi
echo "Done: $(gh repo view --json url -q .url)"
