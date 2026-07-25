# HEARTBEAT.md - Write to Daily Log

## MANDATORY: Write to memory/YYYY-MM-DD.md EVERY heartbeat

Every heartbeat cycle, append to today's memory file. No exceptions.

### Steps:

1. Append to memory/YYYY-MM-DD.md (today UTC) — BOTH local AND VPS.

2. Mirror to VPS via paramiko (same content as local).

### Rules:
- Every heartbeat = every write. No skipping.
- Write to BOTH local and VPS memory files.
- Keep under 5 lines.
- Nothing happened? Write All quiet. Still write.
- Errors? Write them.

