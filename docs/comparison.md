# Framework comparison: Customer Onboarding Assistant

_Generated 2026-08-11 06:53 UTC — 5 fixtures × 4 frameworks._

## Environment

| Setting | Value |
| --- | --- |
| profile | `ollama` |
| base_url | `(unset)` |
| model | `(unset)` |
| api_key | `(unset)` |
| LLM configured | **no** — drafting steps skipped |
| PII engine | `presidio` |

> No LLM endpoint is configured, so no welcome email was drafted. Everything decided by policy — validation, masking, injection defense, risk, planning, the task list and the escalation gate — still ran, and is compared below.

## Capability matrix

| Capability | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| Multi-step | yes | no | yes | yes |
| Conditional branching | yes | no | yes | no |
| Tools | yes | yes | no | yes |
| Agents | multi | single | multi | multi |

- **maf** — Typed executors and edges, checked when the workflow is built rather than when it runs. Switch-case edge groups make one-of-N routing explicit rather than a chain of ifs buried in a node.
- **langchain** — The agent picks its own tool order, so the control flow only exists at runtime — there is no topology to inspect before the run, only a transcript to read after it.
- **langgraph** — Explicit graph: the control flow is data, so every branch can be inspected before the run rather than discovered during it. The repair loop is a real cycle in the graph, not a retry in a wrapper.
- **crew** — The only multi-agent adapter: a copywriter drafts and a separate compliance reviewer checks, each with its own context window. The order is fixed at build time and the crew cannot branch, so conditional routing lives in the adapter rather than in the framework.

## Results by fixture

### `enterprise_high_value`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | _error_ | _error_ | _error_ | _error_ |
| risk band | _error_ | _error_ | _error_ | _error_ |
| findings | _error_ | _error_ | _error_ | _error_ |
| PII entities | _error_ | _error_ | _error_ | _error_ |
| injection | _error_ | _error_ | _error_ | _error_ |
| plan strategy | _error_ | _error_ | _error_ | _error_ |
| rule tasks | _error_ | _error_ | _error_ | _error_ |
| llm tasks | _error_ | _error_ | _error_ | _error_ |
| violations | _error_ | _error_ | _error_ | _error_ |
| confidence | _error_ | _error_ | _error_ | _error_ |
| email words | _error_ | _error_ | _error_ | _error_ |
| prompt versions | _error_ | _error_ | _error_ | _error_ |
| llm calls | _error_ | _error_ | _error_ | _error_ |
| duration ms | _error_ | _error_ | _error_ | _error_ |

- **maf failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the MAF workflow reached an executor that needs a model)`
- **langchain failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`
- **langgraph failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the LangGraph adapter reached a node that needs a model)`
- **crew failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`

**Not compared** — fewer than two frameworks produced a result (see the errors above).

### `injection_attempt`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | escalated | escalated | escalated | escalated |
| risk band | medium | medium | medium | medium |
| findings | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE |
| PII entities | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER |
| injection | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION |
| plan strategy | standard | standard | standard | standard |
| rule tasks | 8 | 8 | 8 | 8 |
| llm tasks | 0 | 0 | 0 | 0 |
| violations | — | — | — | — |
| confidence | 0.0 | 0.0 | 0.0 | 0.0 |
| email words | — | — | — | — |
| prompt versions | — | — | — | — |
| llm calls | 0 | 0 | 0 | 0 |
| duration ms | 381 | 372 | 386 | 397 |

**IDENTICAL** — all frameworks agree on every deterministic field.

### `invalid_missing_fields`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | escalated | escalated | escalated | escalated |
| risk band | high | high | high | high |
| findings | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE |
| PII entities | EMAIL_ADDRESS | EMAIL_ADDRESS | EMAIL_ADDRESS | EMAIL_ADDRESS |
| injection | — | — | — | — |
| plan strategy | remediation | remediation | remediation | remediation |
| rule tasks | 1 | 1 | 1 | 1 |
| llm tasks | 0 | 0 | 0 | 0 |
| violations | — | — | — | — |
| confidence | 0.0 | 0.0 | 0.0 | 0.0 |
| email words | — | — | — | — |
| prompt versions | — | — | — | — |
| llm calls | 0 | 0 | 0 | 0 |
| duration ms | 563 | 349 | 374 | 365 |

**IDENTICAL** — all frameworks agree on every deterministic field.

### `pii_heavy`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | _error_ | _error_ | _error_ | _error_ |
| risk band | _error_ | _error_ | _error_ | _error_ |
| findings | _error_ | _error_ | _error_ | _error_ |
| PII entities | _error_ | _error_ | _error_ | _error_ |
| injection | _error_ | _error_ | _error_ | _error_ |
| plan strategy | _error_ | _error_ | _error_ | _error_ |
| rule tasks | _error_ | _error_ | _error_ | _error_ |
| llm tasks | _error_ | _error_ | _error_ | _error_ |
| violations | _error_ | _error_ | _error_ | _error_ |
| confidence | _error_ | _error_ | _error_ | _error_ |
| email words | _error_ | _error_ | _error_ | _error_ |
| prompt versions | _error_ | _error_ | _error_ | _error_ |
| llm calls | _error_ | _error_ | _error_ | _error_ |
| duration ms | _error_ | _error_ | _error_ | _error_ |

- **maf failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the MAF workflow reached an executor that needs a model)`
- **langchain failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`
- **langgraph failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the LangGraph adapter reached a node that needs a model)`
- **crew failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`

**Not compared** — fewer than two frameworks produced a result (see the errors above).

### `valid_smb`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | _error_ | _error_ | _error_ | _error_ |
| risk band | _error_ | _error_ | _error_ | _error_ |
| findings | _error_ | _error_ | _error_ | _error_ |
| PII entities | _error_ | _error_ | _error_ | _error_ |
| injection | _error_ | _error_ | _error_ | _error_ |
| plan strategy | _error_ | _error_ | _error_ | _error_ |
| rule tasks | _error_ | _error_ | _error_ | _error_ |
| llm tasks | _error_ | _error_ | _error_ | _error_ |
| violations | _error_ | _error_ | _error_ | _error_ |
| confidence | _error_ | _error_ | _error_ | _error_ |
| email words | _error_ | _error_ | _error_ | _error_ |
| prompt versions | _error_ | _error_ | _error_ | _error_ |
| llm calls | _error_ | _error_ | _error_ | _error_ |
| duration ms | _error_ | _error_ | _error_ | _error_ |

- **maf failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the MAF workflow reached an executor that needs a model)`
- **langchain failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`
- **langgraph failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the LangGraph adapter reached a node that needs a model)`
- **crew failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`

**Not compared** — fewer than two frameworks produced a result (see the errors above).

## Verdict

All frameworks produced identical deterministic outcomes on every compared fixture (2 of 5). Any remaining difference is in LLM-authored prose, which is compared structurally rather than verbatim.

Not compared (fewer than two frameworks produced a result): `enterprise_high_value`, `pii_heavy`, `valid_smb`.
