# AgenticAI-101 — Customer Onboarding Assistant, four frameworks, one core

A **Customer Onboarding Assistant** that validates a new customer record, checks
for a duplicate, registers them, mails the customer and the support team, and
then answers questions about who has been onboarded — implemented four times
over a **single shared core**:

| | Microsoft Agent Framework | LangChain | LangGraph | CrewAI |
|---|---|---|---|---|
| Shape | executor graph + tools | one tool-using agent | multi-step graph | two-agent crew |
| Branching | 2 switch-case groups + conditional edges | none (agent decides at runtime) | 3 conditional branch points | none (fixed sequence) |
| HITL | `ctx.request_info()` | blocks, cannot resume | `interrupt()` | blocks, cannot resume |
| State | `FileCheckpointStorage` | **none, by design** | `AsyncSqliteSaver` | **none** (`memory=False`) |
| Resume across processes | yes | **no** | yes | **no** |
| LLM agent identities | 1 | 1 | 0 (plain calls) | **2** |

Because all four call the same schemas, rules and pipeline, any difference in
output is attributable to the *framework* — not to the business logic. That is
the entire point.

> **The comparison is enforced, not asserted.** `tests/integration/test_architecture.py`
> walks the AST of every adapter and fails the build if one grows its own
> thresholds, regexes or business rules. `test_three_way_parity.py` asserts they
> produce byte-identical deterministic outcomes.

---

## Quick start

```bash
uv sync --extra dev --extra nlp     # nlp = the spaCy model Presidio uses
uv run onboarding doctor            # check the environment, no model needed
uv run pytest                       # 583 tests, no API key required
```

### The web page

```bash
uv run onboarding serve             # http://127.0.0.1:8000
```

One page: fill in a customer, pick one of the four frameworks, submit. The agent
validates the record, checks for a duplicate, registers them, writes the welcome
email and sends the task list to the support address — and the page shows you
each message it produced and whether it was actually transmitted. Then the chat
panel opens underneath, backed by the customer registry.

Add `--send` to transmit for real; see [Real mail](#real-mail) first, because
nothing is sent until an address is explicitly approved.

### Or one command per agent, in the terminal

```bash
uv run onboarding demo --framework maf
uv run onboarding demo --framework langchain
uv run onboarding demo --framework langgraph
uv run onboarding demo --framework crew
```

Each onboards a customer, prints the draft, and drops you into a conversation
about the customer it just processed. `onboarding` with no arguments does the
same thing with defaults.

### Point it at a model (free and local)

Everything above runs without a model. To draft emails you need one
OpenAI-compatible endpoint — the same config serves all four frameworks:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b-instruct
ollama serve

cp .env.example .env                # already set up for Ollama
uv run onboarding run --framework langgraph --record fixtures/customers/valid_smb.json
```

### Any OpenAI-compatible provider

Nothing in the model wiring names a provider — `core/llm.py` only ever passes
`base_url`, `model` and `api_key` through. **Switching provider is three
environment variables and no code change.** `.env.example` carries ready-made
profiles:

| Profile | Endpoint | Notes |
|---|---|---|
| **ollama** (default) | `localhost:11434/v1` | free, fully local, no key |
| **groq** | `api.groq.com/openai/v1` | free tier, very fast; pick a tool-calling model |
| **gemini** | `generativelanguage.googleapis.com/v1beta/openai/` | free tier is tight — 5/min, 20/day per model |
| **anthropic** | `api.anthropic.com/v1/` | paid |

Because each chat question costs two calls (decide the tool, then answer),
Gemini's free tier runs out quickly — Groq or Ollama are the better choice for
extended use.

Two provider quirks worth knowing, both found the hard way:

- **Gemini 3.x cannot do tool calling here.** It requires a `thought_signature`
  on function-call parts that the OpenAI-compatible clients don't send, so it
  fails with a 400. Gemini **2.5** works fine.
- **Use a Chat Completions model**, not a Responses-API one — Ollama and most
  local servers only implement `/v1/chat/completions`.

**There is no stub or fake model anywhere in `src/`.** A missing endpoint raises
`LlmNotConfiguredError` with instructions. A test enforces this: a silent
fallback model would make the whole comparison meaningless.

---

## What it does

```
Perception  →  Planning  →  Action  →  Reflection
```

1. **Perception** — validate the record, mask PII with Presidio *before any text
   reaches a model*, scan free text for prompt injection, chunk long notes.
2. **Planning** — score risk, decompose the work least-to-most, rewrite the task
   into an explicit instruction.
3. **Action** — draft the welcome email (the only unavoidable LLM call) and build
   the internal task list.
4. **Reflection** — run the output validators, repair once, then escalate or ship.

A **human-in-the-loop checkpoint** sits before the email is finalised. If the
record is high-risk — enterprise tier, contract value over threshold, injection
detected, or a blocking validation error — the run **pauses, writes an
`approval_required` audit entry, marks the record BLOCKED and stops**. No email
is drafted and no tokens are spent.

```bash
uv run onboarding run --framework langgraph --record fixtures/customers/enterprise_high_value.json
# -> blocked_awaiting_approval, prints a run_id

uv run onboarding pending          # what's waiting on a human

# ...later, from a completely different process:
uv run onboarding resume --run-id <id> --decision approve --by alex
```

The LangChain adapter blocks identically but **cannot** be resumed — it has no
checkpointer and no thread to return to. That is the stateless/stateful
distinction made concrete rather than described.

---

## Registering the customer, and talking to the agent afterwards

The full workflow is: **validate → check for a duplicate → register → mail the
team and the customer → answer questions about them.**

```bash
uv run onboarding run --framework langgraph --record fixtures/customers/valid_smb.json
uv run onboarding registry show          # the CSV table, phone numbers masked
uv run onboarding outbox                 # the mail that was produced
uv run onboarding chat --framework maf   # ask questions about who's onboarded
```

### The registry

A plain CSV at `.runs/registry.csv` — `name, email, phone, plan`, plus the
bookkeeping to trace a row back to its run. Open it in Excel whenever you like;
writes take an exclusive lock and re-check inside it, so a run cannot clobber
another. `onboarding registry export --out anywhere.csv` copies it out.

Plans are **free, pro, pro+**. A record can set `plan` explicitly; otherwise it
derives from `tier` (starter→free, growth→pro, enterprise→pro+), so the
onboarding policy keeps speaking tiers while the business speaks plans.

**Duplicate detection** rides the machinery that already exists rather than
adding a branch. Same `record_id` or email is a *blocking error* → the run
escalates and never re-registers. Same company or shared email domain is a
*warning* → it routes to the human approval gate, because a second team at the
same company is a legitimate thing to onboard and that call belongs to a person.

A customer is written to the registry **only** on a run that completed cleanly:
blocked, rejected, escalated and failed-reflection runs all leave the table
untouched, checked explicitly rather than inferred from call order.

### Task lists

Each customer's onboarding checklist is its own CSV at
`.runs/tasks/<record_id>.csv`, pointed at by a `tasks_path` column on the
registry row. Every task starts `pending`; `core/tasks.mark()` flips one to
`completed`. That file is what makes *"how many tasks are done for Ada?"* an
answerable question — the count is computed in Python and handed to the model as
a finished sentence, because tallying rows is exactly what a small model gets
subtly wrong.

### Real mail

Nothing is transmitted by default. Every message is written as a real `.eml`
file to `.runs/outbox/` and logged. Transmitting one needs **all** of:

1. `SMTP_HOST` configured (Gmail needs an [App Password](https://myaccount.google.com/apppasswords),
   not your account password — see `.env.example`),
2. the `--send` flag, and
3. for anything customer-facing, the recipient on `ONBOARDING_ALLOWED_RECIPIENTS`.

That third condition is the one that matters. A demo form accepts whatever
address is typed into it, and mail to a stranger cannot be recalled — so the
allowlist is **empty by default**, which means no customer mail goes anywhere
until you name an address. Setting it to `*` disables the check, and is
deliberately awkward to type by accident.

Internal task lists all go to one `ONBOARDING_SUPPORT_EMAIL`.

The welcome email is drafted against `<PERSON_1>` placeholders; the real values
are substituted back in deterministic code at the last moment before delivery.

**The duplicate path still mails.** A returning customer gets a short "you
already have an account" note — deterministic text, no model involved, because
there is nothing to draft and a known customer should get the same message every
time.

### Chat — read-only, all four frameworks

```bash
uv run onboarding chat -f langchain                     # a conversation
uv run onboarding chat -f maf --ask "how many are on pro+?"
uv run onboarding chat -f crew --ask "how many tasks are done for Ada?"
uv run onboarding chat -f langgraph -r fixtures/customers/valid_smb.json
```

Same system prompt, same tools, four orchestrations: LangChain's native agent
loop, a LangGraph `agent ↔ tools` graph, a MAF `Agent` with the tools attached,
and a CrewAI crew rebuilt per question — a crew is built to finish a task and
stop, so it is the one framework here with no native turn loop.

Two properties are enforced by tests, not by prompt wording:

- **The model cannot write.** Every tool it can reach is a pure query. There is
  no path from any tool to the registry's write path, the mailer or the approval
  store — `test_chat_readonly.py` walks each tool's call graph and fails the
  build if one appears. Ask it to add a customer and it will tell you it can't.
- **It never has a phone number.** Not masked — *absent*. `VisibleCustomer` has
  no phone field at all, so there is no code path that could surface one and
  nothing for a jailbreak to reach. Names, companies, plans and email addresses
  are ordinary business facts an employee may see, and pass through in full.

## Business rules are enforced, not requested

### No fabricated discounts

Three layers, and only the middle one actually counts:

1. **Preventive** — the prompt states exactly which concessions are approved
   (`render_allowlist`), or says NONE.
2. **Detective** — `DiscountClaimValidator` scans generated prose for
   percentages, money-off, free periods, waivers and promotional language, and
   rejects any claim not present in the record's `declared_discounts`. Numbers
   quoted from the record ("your 12-month term") are whitelisted first.
3. **Corrective** — one repair attempt, then the offending sentences are
   **deleted outright**.

The invariant `validate(redact(text)) == []` is asserted for every adversarial
phrasing in the suite, so an undeclared discount cannot reach a customer even if
the model insists on one.

### PII masking

Microsoft Presidio (free, local, no cloud call) masks emails, phones, names,
cards, IBANs, IPs and locations into stable `<PERSON_1>` placeholders **before**
the prompt is built. Two engines share one interface: full Presidio with spaCy
NER, and a regex-only fallback seeded with the record's own contact names for
when the spaCy model is unavailable. Masking never silently switches off, and
the active engine is reported in every run.

### Prompt injection

Free-text notes are attacker-controlled. The scanner catches instruction
overrides, role reassignment, system impersonation, policy bypass, exfiltration
attempts and unauthorised-discount demands — through unicode obfuscation and
base64 wrapping. A blocking hit raises risk and forces the record to a human;
the poisoned text never reaches a model at all.

### Versioned prompt library

`prompts/*.json`, pinned by `prompts/index.json`, each carrying a checksum over
its semantic content. Editing a prompt without re-checksumming is a hard
failure, so prompt drift is always a deliberate, reviewable change.

```bash
uv run onboarding prompts verify
uv run onboarding prompts rechecksum   # after an intentional edit
```

---

## Comparing the four

```bash
uv run onboarding compare
```

Runs every framework over every fixture and writes [`docs/comparison.md`](docs/comparison.md):
a capability matrix, a per-fixture side-by-side of every field, an explicit
divergence report, and a structural (never verbatim) prose comparison.

Prose is compared on invariants — greeting present, company named, no discount
tokens, word count in band, no PII — plus a deliberately loose content-word
overlap floor. Asserting identical wording across frameworks would be asserting
something untrue.

---

## Layout

```
src/onboarding/
  core/          # ← all business logic lives here, and only here
    schemas.py       shared input + identical output schema
    rules.py         thresholds, tone policy, task templates
    pii.py           Presidio + regex fallback
    injection.py     prompt-injection defenses
    discounts.py     the no-fabricated-discounts guarantee
    registry.py      the CSV customer table (queries vs the single write path)
    qa.py            read-only question answering, masking applied
    mailer.py        outbox by default, SMTP strictly opt-in
    tasks.py         one checklist CSV per customer, with completion status
    steps.py         perceive / plan / act / reflect / register / notify
    hitl.py          pause, log, stop, resume
  adapters/
    maf/           Microsoft Agent Framework workflow + tools
    lc/            the single LangChain agent
    lg/            the LangGraph graph
    crew/          the two-agent CrewAI crew
  chat/          the read-only Q&A agent, in all four frameworks
  web/           the localhost demo page (FastAPI + one HTML file)
  cli/           doctor, serve, demo, run, resume, pending, chat, registry,
                 outbox, compare, concepts, prompts, audit
```

Every adapter node is a wrapper: unpack state → call one `core` function → pack
state. Nothing else.

---

## Concept map

Every principle is bound to code with a `@concept` decorator and the mapping is
generated from the registry, so the docs cannot drift:

```bash
uv run onboarding concepts                  # all of them
uv run onboarding concepts --framework maf  # one layer
```

See [`docs/concepts.md`](docs/concepts.md) for the generated table.

---

## Notes and deviations

- **`agent-framework-core` + `agent-framework-openai`**, not the `agent-framework`
  meta-package, which pulls ~30 provider packages for the same API.
- **`OpenAIChatCompletionClient`**, not the Responses-API client — Ollama and
  most local servers only implement `/v1/chat/completions`.
- **`AsyncSqliteSaver`**, not `SqliteSaver`: the sync saver raises on `ainvoke`.
- **`en_core_web_sm` is not on PyPI** — it installs from a GitHub release wheel
  via the `nlp` extra. Without it the regex PII engine takes over.
- `adapters/maf/executors.py` deliberately omits `from __future__ import annotations`:
  `@response_handler` validates the raw `inspect.signature` annotation, so
  postponed annotations make it reject a valid `WorkflowContext`.

## Testing

```bash
uv run pytest              # 583 model-free tests
uv run pytest -m llm       # 38 more, needs an endpoint (skips without one)
```
