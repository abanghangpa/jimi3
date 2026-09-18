# SELF-IMPROVEMENT PROTOCOL

## Purpose

Make the agent measurably better over time — not just in theory, but in practice.

This protocol turns the abstract "learn from mistakes" instruction into a concrete, trackable system.

---

## 1. Error & Lesson Tracker

File: `memory/lessons.md`

Every time something fails or a better approach is discovered, log it:

```markdown
## YYYY-MM-DD — [Short Title]

**Context:** What was being attempted
**Failure:** What went wrong (or what was suboptimal)
**Root Cause:** Why it happened
**Fix:** What resolved it
**Category:** [tool|skill|memory|config|approach|external]
**Recurring:** [yes|no]
**Status:** [open|resolved|promoted-to-skill]
```

Rules:
- Log failures immediately after resolution
- Mark recurring issues — these become skill updates or new procedures
- Promote resolved recurring issues to Skills
- Review weekly (or on heartbeat) — close stale items

---

## 2. Success Pattern Tracker

File: `memory/successes.md`

When a procedure works exceptionally well, log it:

```markdown
## YYYY-MM-DD — [Short Title]

**What worked:** Description
**Why it worked:** Key factors
**Reusable:** [yes|no]
**Category:** [approach|tool-use|decomposition|research|execution]
**Status:** [noted|promoted-to-skill]
```

This prevents the bias of only learning from failures.

---

## 3. Procedure Effectiveness

For any Skill or repeated procedure, track:

```markdown
## [Skill Name] — Effectiveness Log

| Date | Task | Outcome | Time | Notes |
|------|------|---------|------|-------|
| YYYY-MM-DD | ... | ✅ success / ❌ fail / ⚠️ partial | Xm | ... |
```

After 3+ uses, evaluate:
- Success rate < 70% → revise or retire
- Success rate > 90% and frequently used → consider optimizing
- Never used after 30 days → consider retiring

---

## 4. Periodic Review Cadence

### After Every Meaningful Session
- Update daily memory log
- Log any failures or lessons
- Log any notable successes

### Weekly (or on heartbeat trigger)
- Review `memory/lessons.md` — close resolved items, promote recurring ones
- Review `memory/successes.md` — check for promotable patterns
- Review MEMORY.md — remove stale entries, consolidate duplicates
- Check skill effectiveness logs

### Monthly
- Full skill audit — are all skills still relevant and working?
- Memory consolidation — merge, prune, promote
- Review this protocol itself — is the improvement system working?

---

## 5. Improvement Metrics

Track in `memory/metrics.md`:

```markdown
## Metrics — YYYY-MM-DD

### Skills
- Total active: X
- Created this period: X
- Retired this period: X
- Average effectiveness: X%

### Memory
- Durable entries: X
- Lessons logged: X
- Lessons resolved: X
- Recurring issues: X

### Sessions
- Tasks completed: X
- Tasks partially completed: X
- Tasks blocked: X
- Self-improvement actions taken: X
```

Update monthly or when requested.

---

## 6. Anti-Patterns

Do NOT:
- Log every trivial action
- Create metrics for the sake of metrics
- Let the improvement system become overhead that slows actual work
- Optimize for number of skills (optimize for reuse × reliability × value)
- Turn memory into a transcript

DO:
- Keep logs concise and actionable
- Focus on patterns, not one-offs
- Let the system evolve — if a tracking method isn't useful, change it
- Prefer a single well-maintained skill over five unused ones

---

## 7. Automated Learning Loop

The learning loop must fire automatically — not rely on the agent remembering.

### Trigger: After Complex Tasks

A task is "complex" when it involved:
- 5+ tool calls
- A failure followed by a fix
- Research → execution → verification cycle
- Multiple approaches tried before success
- External API integration or debugging

### Auto-Learning Procedure

After any complex task completes:

1. **Failure scan** — did anything fail or need retry?
   - Yes → log to `memory/lessons.md` with root cause + fix
   - No → skip

2. **Success scan** — did a procedure work unusually well?
   - Yes → log to `memory/successes.md`
   - No → skip

3. **Reusability check** — would this procedure be useful again?
   - Yes → check if a skill exists for it
     - Skill exists → update effectiveness log
     - No skill → **AUTO-CREATE** a skill immediately using the `/learn` procedure (skills/learn/SKILL.md). Do NOT wait for user to ask. Read the relevant source files, distill the procedure, and write the skill to `skills/<name>/SKILL.md`.
   - No → skip

4. **Memory check** — did I learn something durable?
   - Yes → promote to MEMORY.md (respecting curation limits)
   - No → skip

5. **Daily log** — write session summary to `memory/YYYY-MM-DD.md`

### What NOT to auto-learn
- Trivial single-tool lookups
- Simple Q&A that didn't involve execution
- Routine message replies
- Information already in project files

## 8. The Self-Improvement Question

After every meaningful task, ask:

> **Would this have gone better if I had a skill/memory/lesson for it?**

If yes → create or update the relevant artifact.
If no → it was just context-specific, move on.

This single question drives the entire improvement loop.

### Enforcement

This question is MANDATORY after complex tasks (see §7). Do not skip it.
The agent should not need the user to remind it to learn.
