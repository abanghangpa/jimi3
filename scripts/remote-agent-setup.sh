#!/bin/bash
# remote-agent-setup.sh
# Run this after SSHing into srv1686419 to set up your workspace
# with the same protocols, thinking framework, and skills as the primary agent.
#
# Usage: bash /root/.openclaw/workspace/scripts/remote-agent-setup.sh [YOUR_WORKSPACE_PATH]
#
# This copies the essential .md files to your local workspace so you
# can follow the same analytical framework and handover protocol.

set -e

VPS_WORKSPACE="/root/.openclaw/workspace"
REMOTE_WORKSPACE="${1:-$HOME/.openclaw/workspace}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         REMOTE AGENT SETUP — srv1686419                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Source:      $VPS_WORKSPACE"
echo "Destination: $REMOTE_WORKSPACE"
echo ""

# Create directories
mkdir -p "$REMOTE_WORKSPACE/skills"
mkdir -p "$REMOTE_WORKSPACE/memory"

# ─── CORE IDENTITY & OPERATING FILES ───────────────────────
echo "▸ Copying core operating files..."
for f in AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md THINKING.md; do
    if [ -f "$VPS_WORKSPACE/$f" ]; then
        cp "$VPS_WORKSPACE/$f" "$REMOTE_WORKSPACE/$f"
        echo "  ✓ $f"
    fi
done

# ─── MEMORY & LEARNING FILES ───────────────────────────────
echo "▸ Copying memory & learning files..."
for f in MEMORY.md MEMORY-ARCHITECTURE.md SELF-IMPROVEMENT-PROTOCOL.md; do
    if [ -f "$VPS_WORKSPACE/$f" ]; then
        cp "$VPS_WORKSPACE/$f" "$REMOTE_WORKSPACE/$f"
        echo "  ✓ $f"
    fi
done

# ─── JIMI FRAMEWORK FILES ──────────────────────────────────
echo "▸ Copying JIMI framework files..."
for f in JIMI_FRAMEWORK.md JIMI-MARKET-ANALYSIS.md REPORTING.md OPTIMIZATION_FRAMEWORK.md STRATEGY_PROTOCOLS.md; do
    if [ -f "$VPS_WORKSPACE/$f" ]; then
        cp "$VPS_WORKSPACE/$f" "$REMOTE_WORKSPACE/$f"
        echo "  ✓ $f"
    fi
done

# ─── SKILLS ────────────────────────────────────────────────
echo "▸ Copying skills..."
if [ -d "$VPS_WORKSPACE/skills" ]; then
    cp -r "$VPS_WORKSPACE/skills/"* "$REMOTE_WORKSPACE/skills/" 2>/dev/null || true
    echo "  ✓ skills/ ($(ls "$REMOTE_WORKSPACE/skills/" | wc -l) skills)"
fi

# ─── SYNC PROTOCOL ─────────────────────────────────────────
echo "▸ Copying sync protocol..."
for f in SYNC-PROTOCOL.md; do
    if [ -f "$VPS_WORKSPACE/$f" ]; then
        cp "$VPS_WORKSPACE/$f" "$REMOTE_WORKSPACE/$f"
        echo "  ✓ $f"
    fi
done

# ─── TODAY'S MEMORY ────────────────────────────────────────
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d 2>/dev/null || echo "")
echo "▸ Copying recent memory files..."
for d in "$TODAY" "$YESTERDAY"; do
    if [ -f "$VPS_WORKSPACE/memory/$d.md" ]; then
        cp "$VPS_WORKSPACE/memory/$d.md" "$REMOTE_WORKSPACE/memory/$d.md"
        echo "  ✓ memory/$d.md"
    fi
done

# ─── SUMMARY ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                        ║"
echo "║  Your workspace now has:                               ║"
echo "║  • AGENTS.md      — operating rules + remote protocol  ║"
echo "║  • THINKING.md    — analytical framework (18 rules)    ║"
echo "║  • SOUL.md        — identity & philosophy              ║"
echo "║  • skills/        — JIMI market analysis + /learn      ║"
echo "║  • MEMORY.md      — durable knowledge                  ║"
echo "║  • JIMI docs      — framework, strategies, reporting   ║"
echo "║                                                        ║"
echo "║  READ THESE BEFORE STARTING WORK:                      ║"
echo "║  1. THINKING.md — how to think                         ║"
echo "║  2. AGENTS.md §19 — remote agent protocol              ║"
echo "║  3. skills/jimi-market-analysis/SKILL.md — if JIMI     ║"
echo "║                                                        ║"
echo "║  ON COMPLETION (MANDATORY — write to THIS VPS):        ║"
echo "║  • Daily log → /root/.openclaw/workspace/memory/YYYY-MM-DD.md ║"
echo "║  • Lessons  → /root/.openclaw/workspace/memory/lessons.md     ║"
echo "║  • Skills   → /root/.openclaw/workspace/skills/               ║"
echo "║  • Update   → /root/.openclaw/workspace/MEMORY.md if durable  ║"
echo "║  • Commit code changes to git                                 ║"
echo "║                                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Workspace ready at: $REMOTE_WORKSPACE"
echo "Start working!"
