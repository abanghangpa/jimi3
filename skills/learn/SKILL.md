# /learn — Turn Any Source Into a Skill

## Purpose

Convert any source — a URL, a file, a book, a conversation, pasted notes, or a described procedure — into a reusable Skill. This is the agent's primary knowledge ingestion capability.

## When to Use

- User says "learn this", "turn this into a skill", or provides a source to ingest
- After complex tasks that would benefit from a reusable procedure (auto-trigger)
- When the agent encounters documentation that should be captured as procedural knowledge

## Sources Supported

| Source Type | How to Ingest |
|-------------|---------------|
| URL / web page | Fetch with `web_extract`, distill into skill |
| Local file / directory | Read with `read` or `exec`, extract procedure |
| Book / PDF / large docs | Chunk into knowledge-base skill with references/ |
| Conversation | Extract the procedure that was demonstrated |
| Pasted notes / described procedure | Structure into skill format |
| Existing code / script | Analyze and document the procedure |

## Procedure

### Step 1 — Identify Source Type

Determine what kind of source the user provided. Route to the appropriate ingestion path.

### Step 2 — Gather Material

- **URL:** Fetch with `web_extract(url, maxChars=15000)`. If truncated, fetch remaining.
- **File/Dir:** Read contents. For directories, list structure then read key files.
- **Book/PDF:** Use `pdf` tool or extract text. Create knowledge-base skill with per-chapter references.
- **Conversation:** Review recent context for demonstrated procedures.
- **Notes:** Parse the user's description directly.
- **Code:** Read and analyze the script/module.

### Step 3 — Distill Knowledge

Extract from the source:
- **Core procedure** — the repeatable steps
- **Mental models** — how to think about this domain
- **Key decisions** — what choices were made and why
- **Gotchas** — common mistakes, edge cases, prerequisites
- **Anti-patterns** — what NOT to do

Do NOT reproduce source text verbatim. Synthesize and compress.

### Step 4 — Draft the Skill

Create at `skills/<skill-name>/SKILL.md`:

```markdown
# [Skill Name]

## Purpose
One sentence: what this skill does and when to use it.

## Version History
| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 0.1 | YYYY-MM-DD | Created via /learn | Source: [description] |

## Prerequisites
- Tools, access, knowledge required

## Procedure

### Step 1 — [Action]
What to do and why.

### Step 2 — [Action]
What to do and why.

### Step N — Verify
How to confirm success.

## Key Concepts
- Mental models, domain knowledge, important distinctions

## Known Limitations
- What this skill does NOT handle
- Edge cases

## Anti-Patterns
- Common mistakes to avoid

## Usage Log
| Date | Task | Result | Notes |
|------|------|--------|-------|
| YYYY-MM-DD | [original source] | ✅ | Created via /learn |
```

### Step 5 — Handle Large Sources

For books, papers, large docs:
1. Create a lean SKILL.md with core mental models + index
2. Create `references/` directory with one file per chapter/topic
3. Reference files are loaded on demand — not injected into context
4. Include a glossary or cheatsheet when useful

### Step 6 — Log and Report

1. Log creation in `memory/auto-created-skills.md`
2. Report to user:
   - What was created
   - Key procedures captured
   - Where the skill lives
   - Any limitations or gaps

## Skill Naming

- lowercase, hyphens for spaces
- verb-noun format preferred
- descriptive but concise

Examples: `api-auth-flow/`, `deploy-docker-stack/`, `crypto-market-analysis/`

## Deduplication

Before creating a new skill:
1. Check `skills/` for existing skills on the same topic
2. If one exists → update it with new information instead
3. If partial overlap → create a focused sub-skill and cross-reference

## Knowledge-Base Skills

For large sources (books, papers, specs):

```
skills/<name>/
├── SKILL.md          # Core mental models + index
├── references/
│   ├── chapter-01.md # Distilled per-chapter knowledge
│   ├── chapter-02.md
│   └── glossary.md   # Key terms and definitions
└── examples/         # Code samples, templates if applicable
```

Reference files cost nothing until needed — the agent loads them on demand.