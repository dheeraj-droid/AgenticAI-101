# AgenticAI-101 — Customer Onboarding Assistant, four frameworks, one core

A **Customer Onboarding Assistant** that validates a new customer record, checks
for a duplicate, registers them, mails the customer and the support team, and
then answers questions about who has been onboarded — implemented four times
over a **single shared core**:

| | Microsoft Agent Framework | LangChain | LangGraph | CrewAI |
|---|---|---|---|---|
| Shape | executor graph + tools | one tool-using agent | multi-step graph | two-agent crew |
| Branching | 2 switch-case groups | none (agent decides at runtime) | 2 conditional branch points | none (fixed sequence) |
| Control flow visible before the run | yes (typed edges) | **no** (agent decides) | yes (graph is data) | partly (fixed order) |
| LLM agent identities | 1 | 1 | 0 (plain calls) | **2** |

Because all four call the same schemas, rules and pipeline, any difference in
output is attributable to the *framework* — not to the business logic. That is
the entire point.

> **The comparison is enforced, not asserted.** `tests/integration/test_architecture.py`
> walks the AST of every adapter and fails the build if one grows its own
> thresholds, regexes or business rules. `test_three_way_parity.py` asserts they
> produce byte-identical deterministic outcomes.

---

## Demo it in five minutes

### 1. Install

```bash
git clone https://github.com/dheeraj-droid/AgenticAI-101.git
cd AgenticAI-101
uv sync --extra dev --extra nlp     # nlp = the spaCy model Presidio uses
uv run onboarding doctor            # all four adapters should say ok
```

`doctor` will report **LLM endpoint: not configured** — that is expected, and
everything policy-driven still runs. Step 2 fixes it.

### 2. Point it at a model

Any OpenAI-compatible endpoint drives all four frameworks. Free and local:

```bash
ollama pull qwen2.5:3b-instruct
ollama serve
cp .env.example .env                # already set up for Ollama
```

Or edit `.env` for Groq / Gemini / Anthropic — three variables, no code change.

### 3. Run the page

```bash
uv run onboarding serve             # http://127.0.0.1:8000
```

One tab per framework. Fill in the customer once, then run it through whichever
agent you like — each tab keeps its own result and its own chat, so you can flip
between them and compare. A strip across the top counts the registry: how many
customers, how many on each plan, how many tasks are done.

Each result shows **what that framework actually executed** — MAF lists the
typed executors its edges routed through, LangGraph lists the graph nodes it
entered, LangChain lists the tools its agent chose for itself at runtime, and
CrewAI lists its two agents and the task each of them ran. Same record, same
decision, four visibly different routes. That is the answer to "is this
framework really being used?" — it is not a label, it is a recording.

**Submit the same customer twice.** The second run takes the duplicate branch:
no registration, no model call, and a plain "you already have an account" note
instead of a welcome email.

**Then switch tabs and press Onboard again.** Same form, same pipeline — the only
thing that changed is which agent framework ran it. A dot on a tab means that
framework has a result waiting; flip between them to compare.

Things to ask the chat panel:

```
how many customers are on pro?
how many tasks are done for Ada?
what plan is Northwind Trading on?
what is their phone number?      ← it does not have one, and says so
add a customer called Bob         ← it cannot write, and says so
```

### 4. Real mail with Gmail

Gmail needs an **App Password**, not your account password: turn on 2-Step
Verification, then generate one at
[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).
Paste it with the spaces removed.

```bash
# in .env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=dheerajreddy035@gmail.com
SMTP_PASSWORD=<16-char App Password, spaces removed>
ONBOARDING_FROM_EMAIL=dheerajreddy035@gmail.com
ONBOARDING_SUPPORT_EMAIL=dheerajreddy035@gmail.com
ONBOARDING_ALLOWED_RECIPIENTS=*

uv run onboarding serve --send
```

`ONBOARDING_ALLOWED_RECIPIENTS=*` is the demo setting: the welcome email really
goes to **whatever address was typed into the form**. That is what you want for
a live demo and it is worth being clear-eyed about — a mistyped address reaches
a real stranger's inbox and cannot be recalled. The page shows an amber
*"sending to any address"* badge the whole time it is on.

To be careful instead, list the addresses:
`ONBOARDING_ALLOWED_RECIPIENTS=you@gmail.com,someone@else.com`. Anything else
still lands in the outbox, and the page says plainly that it was refused rather
than implying it went out.

### The same thing in the terminal

```bash
uv run onboarding demo --framework maf
uv run onboarding demo --framework langchain
uv run onboarding demo --framework langgraph
uv run onboarding demo --framework crew
```

Each onboards a customer, prints the draft, and drops you into a conversation
about the customer it just processed. `onboarding` with no arguments does the
same thing with defaults.

### Worth showing off

```bash
uv run onboarding run -f langgraph -r fixtures/customers/injection_attempt.json
#   the poisoned notes never reach a model: escalated, 0 model calls, not registered

uv run onboarding run -f crew -r fixtures/customers/invalid_missing_fields.json
#   a blocking validation error stops the run before drafting, in every framework

uv run onboarding registry show          # the CSV, phone numbers masked
uv run onboarding outbox                 # every message produced
uv run onboarding compare                # all four, side by side
uv run onboarding concepts               # every principle → the code that implements it
uv run pytest                            # 535 tests, no API key required
```

---

## Any OpenAI-compatible provider

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

**One gate sits before drafting.** A record with blocking validation errors, or
one carrying a prompt-injection attempt, escalates: it is never drafted from,
never registered, and costs no model call. Everything else is onboarded
autonomously.

```bash
uv run onboarding run --framework langgraph --record fixtures/customers/injection_attempt.json
# -> escalated, llm_calls 0, registered no
```

Risk scoring still runs, and still shapes the plan and the confidence floor — an
enterprise record gets the `enterprise` strategy and an extra grounding step —
but a high score is not a veto on its own. The two graphs route on one shared
predicate, `OnboardingState.must_escalate`, so they cannot drift apart.

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
*warning* → it is reported but does not stop the run, because a second team at
the same company is a legitimate thing to onboard.

A customer is written to the registry **only** on a run that completed cleanly:
escalated and failed-reflection runs leave the table untouched, checked
explicitly rather than inferred from call order.

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
  no path from any tool to the registry's write path or the mailer —
  `test_chat_readonly.py` walks each tool's call graph and fails the
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
base64 wrapping. A blocking hit escalates the record before drafting, so the
poisoned text never reaches a model at all.

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
  adapters/
    maf/           Microsoft Agent Framework workflow + tools
    lc/            the single LangChain agent
    lg/            the LangGraph graph
    crew/          the two-agent CrewAI crew
  chat/          the read-only Q&A agent, in all four frameworks
  web/           the localhost demo page (FastAPI + one HTML file)
  cli/           doctor, serve, demo, run, chat, registry, outbox,
                 compare, concepts, prompts, audit
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
- **`en_core_web_sm` is not on PyPI** — it installs from a GitHub release wheel
  via the `nlp` extra. Without it the regex PII engine takes over.
- **CrewAI is given `provider="openai"` explicitly.** Without it CrewAI infers a
  provider from the model name, and any name it does not recognise (`qwen2.5:3b`
  and every other local model) falls through to LiteLLM, which is not a
  dependency here.
- **CrewAI telemetry is opted out** before `crewai` is imported. Nothing here
  should be reporting customer records to a third party.

## Testing

```bash
uv run pytest              # 535 model-free tests
uv run pytest -m llm       # 40 more, needs an endpoint (skips without one)
```
