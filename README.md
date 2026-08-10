# AgenticAI-101 — Customer Onboarding Assistant, three frameworks, one core

A **Customer Onboarding Assistant** that validates a new customer record, drafts a
welcome email, generates an internal task list and logs the result — implemented
three times over a **single shared core**:

| | Microsoft Agent Framework | LangChain | LangGraph |
|---|---|---|---|
| Shape | executor graph + tools | one tool-using agent | multi-step graph |
| Branching | 2 switch-case groups + conditional edges | none (agent decides at runtime) | 3 conditional branch points |
| HITL | `ctx.request_info()` | blocks, cannot resume | `interrupt()` |
| State | `FileCheckpointStorage` | **none, by design** | `SqliteSaver` |
| Resume across processes | yes | **no** | yes |

Because all three call the same schemas, rules and pipeline, any difference in
output is attributable to the *framework* — not to the business logic. That is
the entire point.

> **The comparison is enforced, not asserted.** `tests/integration/test_architecture.py`
> walks the AST of every adapter and fails the build if one grows its own
> thresholds, regexes or business rules. `test_three_way_parity.py` asserts the
> three produce byte-identical deterministic outcomes.

---

## Quick start

```bash
uv sync --extra dev --extra nlp     # nlp = the spaCy model Presidio uses
uv run onboarding doctor            # check the environment, no model needed
uv run pytest                       # 229 tests, no API key required
```

### Point it at a model (free and local)

Everything above runs without a model. To draft emails you need one
OpenAI-compatible endpoint — the same config serves all three frameworks:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b-instruct
ollama serve

cp .env.example .env                # already set up for Ollama
uv run onboarding run --framework langgraph --record fixtures/customers/valid_smb.json
```

`.env.example` also carries an Anthropic profile — Anthropic exposes an
OpenAI-compatible `/v1/chat/completions`, so switching providers is three
environment variables and no code change.

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

## Comparing the three

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
    steps.py         the pipeline: perceive / plan / act / reflect
    hitl.py          pause, log, stop, resume
  adapters/
    maf/           Microsoft Agent Framework workflow + tools
    lc/            the single LangChain agent
    lg/            the LangGraph graph
  cli/           doctor, run, resume, pending, compare, concepts, prompts, audit
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
uv run pytest              # 229 model-free tests
uv run pytest -m llm       # 16 more, needs an endpoint (skips without one)
```
