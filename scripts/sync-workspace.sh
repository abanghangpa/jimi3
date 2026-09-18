#!/bin/bash
# sync-workspace.sh — Pull latest from VPS source of truth
# Run this on the other agent's machine before starting work

set -e

WORKSPACE="${1:-$HOME/.openclaw/workspace}"
cd "$WORKSPACE"

echo "=== Workspace Sync ==="
echo "Pulling latest from origin/main..."

# Stash any local changes to source-of-truth files
STASHED=false
if ! git diff --quiet -- AGENTS.md SOUL.md MEMORY.md USER.md TOOLS.md THINKING.md skills/ memory/*.md 2>/dev/null; then
    echo "Stashing local changes to source-of-truth files..."
    git stash push -m "auto-stash before sync" -- AGENTS.md SOUL.md MEMORY.md USER.md TOOLS.md THINKING.md skills/ memory/*.md
    STASHED=true
fi

# Pull with rebase
git pull origin main --rebase

# Pop stash if we stashed
if [ "$STASHED" = true ]; then
    echo "Restoring stashed changes..."
    git stash pop || echo "⚠️  Stash pop had conflicts — resolve manually"
fi

echo ""
echo "=== Sync Complete ==="
echo "Source-of-truth files updated."
echo "Code changes: check git status for any uncommitted work."
echo ""
echo "To push code changes:"
echo "  git add jimi_audit/ stampede/ scripts/"
echo "  git commit -m 'description'"
echo "  git push origin main"
