# Framework comparison: Customer Onboarding Assistant

_Generated 2026-08-11 05:38 UTC — 5 fixtures × 4 frameworks._

## Environment

| Setting | Value |
| --- | --- |
| profile | `ollama` |
| base_url | `(unset)` |
| model | `(unset)` |
| api_key | `(unset)` |
| LLM configured | **no** — drafting steps skipped |
| PII engine | `presidio` |

> No LLM endpoint is configured, so no welcome email was drafted. Everything decided by policy — validation, masking, injection defense, risk, planning, the task list and the approval gate — still ran, and is compared below.

## Capability matrix

| Capability | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| Multi-step | yes | no | yes | yes |
| Conditional branching | yes | no | yes | no |
| HITL pause | yes | yes | yes | yes |
| Durable resume | yes | **no** | yes | **no** |
| Tools | yes | yes | no | yes |
| Agents | multi | single | multi | multi |
| Statefulness | stateful | stateless | stateful | stateless |
| Checkpoint backend | FileCheckpointStorage (.runs/maf_checkpoints) | none (by design) | SqliteSaver (.runs/langgraph.sqlite) | none (memory=False) |

- **maf** — Typed executors and edges. request_info() suspends the workflow and the run resumes via run(responses={request_id: decision}). Switch-case edge groups make one-of-N routing explicit rather than a chain of ifs.
- **langchain** — The agent picks its own tool order, so the control flow only exists at runtime. It can pause for approval but cannot resume: with no checkpointer there is no thread to return to.
- **langgraph** — Explicit graph: the control flow is data, and every branch is inspectable before the run. interrupt() suspends mid-graph and the checkpoint survives process exit.
- **crew** — The only multi-agent adapter: a copywriter drafts and a separate compliance reviewer checks, each with its own context window. The order is fixed at build time and the crew cannot branch, so conditional routing lives in the adapter rather than in the framework.

## Results by fixture

### `enterprise_high_value`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | blocked_awaiting_approval | blocked_awaiting_approval | blocked_awaiting_approval | blocked_awaiting_approval |
| risk band | high | high | high | high |
| needs approval | yes | yes | yes | yes |
| findings | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE |
| PII entities | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER |
| injection | — | — | — | — |
| plan strategy | enterprise | enterprise | enterprise | enterprise |
| rule tasks | 15 | 15 | 15 | 15 |
| llm tasks | 0 | 0 | 0 | 0 |
| violations | — | — | — | — |
| confidence | 0.0 | 0.0 | 0.0 | 0.0 |
| email words | — | — | — | — |
| prompt versions | — | — | — | — |
| llm calls | 0 | 0 | 0 | 0 |
| resume token | CUST-2002-maf-20260811T053810-88540e | **none** | CUST-2002-langgraph-20260811T053811-fcdac8 | **none** |
| duration ms | 436 | 390 | 433 | 508 |

**IDENTICAL** — all frameworks agree on every deterministic field.

### `injection_attempt`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | blocked_awaiting_approval | blocked_awaiting_approval | blocked_awaiting_approval | blocked_awaiting_approval |
| risk band | medium | medium | medium | medium |
| needs approval | yes | yes | yes | yes |
| findings | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE | CONTRACT_START_IN_FUTURE |
| PII entities | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER | EMAIL_ADDRESS, PERSON, PHONE_NUMBER |
| injection | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION | EXFILTRATION, IGNORE_PREVIOUS, ROLE_OVERRIDE, SYSTEM_IMPERSONATION |
| plan strategy | standard | standard | standard | standard |
| rule tasks | 9 | 9 | 9 | 9 |
| llm tasks | 0 | 0 | 0 | 0 |
| violations | — | — | — | — |
| confidence | 0.0 | 0.0 | 0.0 | 0.0 |
| email words | — | — | — | — |
| prompt versions | — | — | — | — |
| llm calls | 0 | 0 | 0 | 0 |
| resume token | CUST-4004-maf-20260811T053813-1fa4d6 | **none** | CUST-4004-langgraph-20260811T053814-203bab | **none** |
| duration ms | 408 | 383 | 422 | 406 |

**IDENTICAL** — all frameworks agree on every deterministic field.

### `invalid_missing_fields`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | escalated | escalated | escalated | escalated |
| risk band | high | high | high | high |
| needs approval | yes | yes | yes | yes |
| findings | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE | GO_LIVE_IN_PAST, IMPLAUSIBLE_DISCOUNT, MALFORMED_CONTACT_PHONE, MISSING_CONTACT_NAME, NO_PRODUCTS, SUSPICIOUS_COMPANY_NAME, TIER_VALUE_MISMATCH, UNAPPROVED_DISCOUNT, ZERO_CONTRACT_VALUE |
| PII entities | EMAIL_ADDRESS | EMAIL_ADDRESS | EMAIL_ADDRESS | EMAIL_ADDRESS |
| injection | — | — | — | — |
| plan strategy | remediation | remediation | remediation | remediation |
| rule tasks | 2 | 2 | 2 | 2 |
| llm tasks | 0 | 0 | 0 | 0 |
| violations | — | — | — | — |
| confidence | 0.0 | 0.0 | 0.0 | 0.0 |
| email words | — | — | — | — |
| prompt versions | — | — | — | — |
| llm calls | 0 | 0 | 0 | 0 |
| resume token | CUST-5005-maf-20260811T053814-cb20cd | **none** | CUST-5005-langgraph-20260811T053815-be96f8 | **none** |
| duration ms | 551 | 358 | 412 | 359 |

**IDENTICAL** — all frameworks agree on every deterministic field.

### `pii_heavy`

| Field | maf | langchain | langgraph | crew |
| --- | --- | --- | --- | --- |
| status | _error_ | _error_ | _error_ | _error_ |
| risk band | _error_ | _error_ | _error_ | _error_ |
| needs approval | _error_ | _error_ | _error_ | _error_ |
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
| resume token | _error_ | _error_ | _error_ | _error_ |
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
| needs approval | _error_ | _error_ | _error_ | _error_ |
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
| resume token | _error_ | _error_ | _error_ | _error_ |
| duration ms | _error_ | _error_ | _error_ | _error_ |

- **maf failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the MAF workflow reached an executor that needs a model)`
- **langchain failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`
- **langgraph failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-... (the LangGraph adapter reached a node that needs a model)`
- **crew failed:** `LlmNotConfiguredError: No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (see .env.example).   Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama   Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...`

**Not compared** — fewer than two frameworks produced a result (see the errors above).

## Verdict

All frameworks produced identical deterministic outcomes on every compared fixture (3 of 5). Any remaining difference is in LLM-authored prose, which is compared structurally rather than verbatim.

Not compared (fewer than two frameworks produced a result): `pii_heavy`, `valid_smb`.
