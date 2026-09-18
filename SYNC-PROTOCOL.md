# SYNC-PROTOCOL.md — Multi-Agent Workspace Sync

## Purpose

Define how multiple OpenClaw agents share and synchronize the workspace at `/root/.openclaw/workspace/` (srv1686419).

## Architecture

```
srv1686419 (THIS VPS)              Other Agent's Machine
┌─────────────────────────┐        ┌─────────────────────────┐
│ /root/.openclaw/workspace│  git   │ ~/.openclaw/workspace   │
│   (source of truth)      │◄──────►│   (clone)               │
│   jimi3.git              │  push  │                         │
│                          │  pull  │                         │
└─────────────────────────┘        └─────────────────────────┘
```

## File Categories

### SOURCE OF TRUTH (srv1686419 only — do NOT edit on clone)

These files are maintained by the primary agent on this VPS. The other agent should pull before reading, never push changes back.

| File | Why |
|------|-----|
| `AGENTS.md` | Operating instructions |
| `SOUL.md` | Identity & philosophy |
| `MEMORY.md` | Durable knowledge |
| `USER.md` | User preferences |
| `TOOLS.md` | Tool notes |
| `IDENTITY.md` | Agent identity |
| `THINKING.md` | Analytical framework |
| `SELF-IMPROVEMENT-PROTOCOL.md` | Learning loop |
| `MEMORY-ARCHITECTURE.md` | Memory rules |
| `SKILL-*.md` | Skill lifecycle docs |
| `SECURITY.md` | Security protocols |
| `WORKFLOW.md` | Workflow rules |
| `REPORTING.md` | Report standards |
| `OPTIMIZATION_FRAMEWORK.md` | Optimization rules |
| `STRATEGY_PROTOCOLS.md` | Strategy definitions |
| `skills/` | All skill definitions |
| `memory/` | Daily logs, lessons, successes |

### SHARED CODE (both agents can edit — use git carefully)

Both agents can modify these. Use branches and merge carefully.

| Directory | What |
|-----------|------|
| `jimi_audit/` | JIMI scanner codebase |
| `stampede/` | STAMPEDE explorer |
| `scripts/` | Utility scripts |

### LOCAL ONLY (never sync)

| File | Why |
|------|-----|
| `*.json` (scan data) | Machine-specific runtime data |
| `*.log` | Local logs |
| `.env` | Secrets (different per machine) |
| `depth_env/` | Virtualenv (rebuild locally) |
| `tmp/` | Temporary files |
| `latest_scan.json` | Runtime output |

## Sync Workflow

### Before starting work (other agent)

```bash
cd ~/.openclaw/workspace
git pull origin main
```

### After making code changes (other agent)

```bash
cd ~/.openclaw/workspace
git add jimi_audit/ stampede/ scripts/
git commit -m "description of change"
git pull origin main --rebase  # rebase on top of any VPS changes
git push origin main
```

### After VPS agent makes changes

```bash
cd /root/.openclaw/workspace
git add <files>
git commit -m "description"
git push origin main
```

### Conflict resolution

If `git pull` produces conflicts:
1. **Source-of-truth files** (AGENTS.md, SOUL.md, skills/, etc.) → accept the VPS version: `git checkout --ours <file>`
2. **Shared code** (jimi_audit/, stampede/) → review both versions, merge manually
3. **Local files** → should never conflict (not in git)

## Rules

1. **Never push source-of-truth files from the clone** — the VPS is authoritative
2. **Always pull before starting work** — get latest framework and skills
3. **Commit code changes with clear messages** — other agent needs to understand what changed
4. **Use branches for experimental work** — merge to main only when validated
5. **Don't commit .env, secrets, or scan data** — these are machine-specific

## SSH Access

The other agent can SSH to this VPS for direct file access when needed:

```bash
ssh root@72.62.73.46
# Workspace: /root/.openclaw/workspace/
```

Use SSH for: reading files, checking status, running quick commands.
Use git for: code changes that need to persist on both machines.