<!--
Thanks for the PR. A wrong verdict matters as much as a crash here, so please
fill in the evidence section rather than only describing the change.
-->

## What & why

<!-- One paragraph. What changes, and why it is needed. Link the issue. -->

Closes #

## Type

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `refactor` — no behaviour change
- [ ] `test` — tests only
- [ ] `chore` / `ci` — tooling, deps, pipeline

## Checklist

- [ ] Conventional Commit title
- [ ] `make gate` is green (format-check → lint → typecheck → security → test)
- [ ] Coverage did not regress (`make coverage`, 55% ratchet)
- [ ] Tests added/updated for the behaviour change
- [ ] Docs updated where behaviour changed (`README.md`, `docs/`, i18n in **all**
      of EN/IT/FR/ES, `.env.example` for new knobs)
- [ ] If a wire format changed: `REPORT_SCHEMA_VERSION` bumped and the Rust
      bridge still round-trips
- [ ] New API fields respect TLP `redact()` / clearance scoping
- [ ] No secrets, no `.env`, no real case evidence in the diff

## Evidence

<!--
Paste the command(s) you ran and their result. "Should work" is not evidence.
e.g. `.venv/bin/python -m pytest tests/test_units.py -q -k my_case` → 5 passed
-->

```text

```

## Risk / rollout

<!-- Blast radius, migration, anything a reviewer should be suspicious about. -->
