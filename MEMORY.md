# MEMORY.md

## Memory Policy

Durable memory should contain high-value information that improves future work.

Use the memory architecture in `MEMORY-ARCHITECTURE.md`.

Do not treat daily memory as permanent memory.

## Durable Knowledge

### Agent Architecture
- The agent architecture separates identity, workspace instructions, security, durable memory, user preferences, daily handover logs, and procedural Skills.
- Daily memory is the session handover layer.
- Skills are procedural memory and should be created or updated only when a procedure is sufficiently repeated or validated.

### Quality Principle
- Prefer high-signal, low-volume memory.
- Preserve uncertainty rather than turning inference into fact.
- Consolidate duplicates and superseded information.

## Promotion Rules

- Session state → daily memory
- Durable fact/lesson → MEMORY.md
- Stable user preference → USER.md
- Repeatable validated procedure → Skill