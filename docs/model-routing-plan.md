# Ringer model routing plan

Ringer must make the model decision before it starts workers. A worker never
inherits a provider or CLI default, because that can silently turn a cheap
batch into a premium run.

## Default lanes

| Lane | Harness and model | Use |
|---|---|---|
| Cheap worker | OpenCode + openrouter/z-ai/glm-5.2 | Mechanical edits, docs, probes, tightly checked work |
| Standard Codex | Codex CLI + gpt-5.5 | Normal implementation, fixes, and reviews |
| Escalation | Explicit task model gpt-5.6-terra or gpt-5.6-sol | Only when the manifest names it and preview/quota use is approved |
| High-volume candidate | Explicit task model gpt-5.6-luna | Only after a bakeoff proves it beats the cheap lane for that task type |

The built-in Codex engine pins gpt-5.5 and passes it with -m. The sample
configuration does the same. This protects fresh installs and local configs
from inheriting a changing CLI default.

## Manifest rules

- Every bakeoff names the exact model in each task's model field.
- A model change is a routing decision, not a worker implementation detail.
- Keep the same scenario and executed check across candidates.
- Record harness, model, access mode, task type, first-try verdict, retry,
  duration, tokens, and estimated API cost or subscription/quota usage.
- Do not promote a preview model to a default from one successful run. Use the
  per-task scoreboard and promotion ladder after representative evidence.

The GPT-5.6 probe template is intentionally lintable but not runnable by
default. Running it requires explicit preview/quota authorization.
