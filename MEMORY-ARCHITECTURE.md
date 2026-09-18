# MEMORY ARCHITECTURE

## Purpose

Memory exists to make future sessions better, not to preserve everything that happened.

The system uses four layers:

```text
Current Context
      ↓
Daily Memory
      ↓
Durable Memory / User Memory
      ↓
Skills
```

Each layer has a different job.

---

## 1. Current Context

Contains what is immediately relevant to the active task.

Do not write everything from context into memory.

Only persist information with future value.

---

## 2. Daily Memory

Location:

`memory/YYYY-MM-DD.md`

Purpose:

- session handover
- recent decisions
- discoveries
- unfinished work
- temporary state
- useful observations

Think of this as the agent's **operational journal**.

It should answer:

- What happened?
- What was decided?
- What was learned?
- What remains?
- What should happen next?

### Daily log format

Prefer:

```markdown
# YYYY-MM-DD

## Work
- ...

## Decisions
- ...

## Discoveries
- ...

## Problems / Lessons
- ...

## Open Loops
- ...

## Next State
- ...
```

Do not turn the daily log into a transcript.

---

## 3. Durable Memory

`MEMORY.md` contains knowledge that remains useful beyond the current session.

Good candidates:

- stable project decisions
- recurring environment facts
- validated workflows
- important lessons
- durable technical constraints
- recurring user working preferences

Bad candidates:

- one-off chatter
- temporary status
- speculative conclusions
- information already obvious from project files
- secrets
- unnecessary sensitive information

---

## 4. User Memory

`USER.md` is specifically about stable preferences and working expectations.

Examples:

- preferred communication style
- recurring output format
- stable workflow preferences
- long-term project goals
- persistent dislikes
- how the user prefers the agent to operate

Do not use USER.md as a biography.

---

## 5. Skills

A Skill is not memory in the ordinary sense.

It is **reusable procedural knowledge**.

Promotion path:

```text
Observation
   ↓
Lesson
   ↓
Repeated / validated procedure
   ↓
Skill
```

Example:

```text
Daily memory:
"Freshdesk migration failed because webhook payloads differed."

MEMORY.md:
"Freshdesk webhook payload differs from the old integration."

Skill:
"Freshdesk migration validation procedure."
```

---

## 6. Promotion Rules

After meaningful work, classify new information.

### Type A — Session state

Write to daily memory.

### Type B — Durable fact

Promote to MEMORY.md.

### Type C — Stable user preference

Promote to USER.md.

### Type D — Repeatable procedure

Create/update a Skill.

### Type E — Secret/private transient data

Do not persist unless explicitly required by the system and securely handled.

---

## 7. Confidence

Memory should preserve epistemic status.

Use labels when useful:

- **FACT** — directly verified
- **OBSERVED** — observed but not independently verified
- **INFERENCE** — reasoned from evidence
- **HYPOTHESIS** — plausible but unconfirmed
- **STALE** — previously valid but may have changed

Do not store an inference as a fact.

---

## 8. Contradictions

When new information conflicts with memory:

1. Verify the new information.
2. Determine whether the old information is stale.
3. Update the durable source of truth.
4. Keep the daily log as historical context when useful.
5. Do not silently preserve both contradictory versions as equal truth.

---

## 9. Garbage Control & Curation Limits

Memory quality degrades when everything is saved.

Prefer:

**high signal / low volume**

### Hard Limits

| File | Limit | Action on overflow |
|------|-------|--------------------|
| `MEMORY.md` | 3,000 chars (~1,100 tokens) | Consolidate before adding |
| `USER.md` | 1,500 chars (~550 tokens) | Merge entries before adding |
| `memory/YYYY-MM-DD.md` | 4,000 chars per day | Summarize older entries |
| Total daily logs | Keep last 30 days | Archive older files |

When a write would exceed the limit:
1. Read current entries
2. Remove or consolidate stale/low-value entries
3. Then add the new entry

### What to remove first
- Duplicates across files
- Obsolete facts (tools changed, env changed)
- Superseded decisions
- Low-value notes ("asked about Python")
- Temporary state that's no longer relevant
- Observations that never proved useful

### What to keep
- Lessons from failures
- Validated procedures
- Stable environment facts
- User preferences that keep coming up
- Decisions with lasting impact

Do not delete historical records merely because they are old if they are needed for audit or project history.

### Maintenance Cadence
- **After each meaningful session:** check if MEMORY.md or USER.md are near limits
- **Weekly:** consolidate daily logs, promote durable facts, remove noise
- **Monthly:** review and prune MEMORY.md for staleness

---

## 10. Handover Standard

At the end of meaningful work, the next agent should be able to continue without reconstructing the entire conversation.

A good handover contains:

```text
Current state
Decision made
Evidence / reason
Outstanding issue
Next action
```

The daily memory is therefore a **handover interface between sessions and agents**.