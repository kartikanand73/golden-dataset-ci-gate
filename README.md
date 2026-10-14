# Golden Dataset CI Gate

A starter framework for regression-testing AI systems in CI. Every change to a model,
prompt or configuration reruns the same locked golden dataset, compares the results
with an approved baseline, and blocks the change if quality or safety regresses.

> **The golden dataset stays fixed. The system changes. The new results are compared
> with the previous baseline results produced from the same dataset.**

## The problem

Changing one model, prompt or tool can change the behaviour of the whole AI system.
An upgrade that looks better on average can still quietly break the few decisions that
matter most, such as routing a security incident to an automated password reset.

## The solution

Treat the model, prompt and configuration as code, and put a gate in front of them:

1. A fixed, versioned **golden dataset** (`data/golden_dataset.jsonl`, 60 cases).
2. An **approved baseline**: the results of the current approved system on that dataset
   (`baselines/`). It only changes through an explicit, human approval.
3. On every commit, GitHub Actions **reruns the candidate** on the same dataset,
   **compares** it decision by decision with the baseline, and applies
   **deterministic gates** (`config/gates.yaml`).
4. A **report** shows the metrics table, why the gate passed or failed, and every
   individual decision that changed.

```
config/model.yaml ─┐                                     ┌─► comparison.md  (job summary)
config/prompt.txt ─┼─► evaluate ─► reports/candidate ─┐  ├─► comparison.json
golden dataset ────┘   (provider)                      ├─► gate ─┼─► changed-cases.jsonl
                                    baselines/ ────────┘  │  └─► junit.xml
                                    gates.yaml ───────────┘      exit 1 on hard-gate failure
```

The sample task is support-ticket routing. Each ticket gets
`{"category": ..., "requires_human": ..., "confidence": ...}` where category is
`PASSWORD_RESET`, `ACCESS_REQUEST`, `SECURITY_INCIDENT` or `GENERAL_SUPPORT`.

## Five-minute demonstration

Requires Python 3.12 and `make`. No API keys, no network calls to model providers.

```bash
make install
make test
make demo-pass    # mock-v1-baseline   -> GOLDEN DATASET GATE: PASS  (exit 0)
make demo-fail    # mock-v2-regression -> GOLDEN DATASET GATE: FAIL  (nonzero exit, reports still written)
make demo-fixed   # mock-v3-fixed      -> GOLDEN DATASET GATE: PASS  (exit 0)
```

Each demo writes `reports/<demo>/comparison/comparison.md`. Expected results:

| Demo | Accuracy | Macro-F1 | Human-review | Critical safety errors | Gate |
|---|---:|---:|---:|---:|---|
| `mock-v1-baseline` | 95.00% | 95.05% | 98.33% | 0 | **PASS** |
| `mock-v2-regression` | 86.67% | 86.30% | 91.67% | 4 | **FAIL** |
| `mock-v3-fixed` | 98.33% | 98.33% | 98.33% | 0 | **PASS** |

The FAIL report names the five security tickets that regressed, four of them flagged
`CRITICAL SAFETY ERROR` because a security incident would have been automated. It also
shows non-blocking warnings: p95 latency up about 38% and cost per 1,000 tickets up
about 60%.

### What the mocks are

The mocks are small deterministic keyword routers, so the demo is reproducible offline.
They never look at case IDs or expected labels; their behaviour comes only from the
ticket text, the way a real model's would.

- **v1** checks for security signals first. It misses three nuanced cases but passes.
- **v2** is a "simplified prompt" that checks password keywords before security
  keywords. Any security ticket mentioning a password gets automated. It is also
  slower, pricier and overconfident.
- **v3** restores security-first ordering, recognises departed-staff incidents and
  stops treating a bare "reset" as a password reset. Two improvements, no regressions.

## GitHub demonstration

1. Create an empty GitHub repository and push this one to it (`main` already contains
   the code and the approved mock-v1 baseline):
   ```bash
   git remote add origin git@github.com:<you>/golden-dataset-ci-gate.git
   git push -u origin main
   ```
   The **Golden Dataset Gate** workflow runs and passes.
2. Change the model in `config/model.yaml`:
   ```yaml
   model: mock-v2-regression
   ```
3. Commit and push (to `main`, or to a branch with a pull request).
4. Watch GitHub Actions fail. The job summary starts with `GOLDEN DATASET GATE: FAIL`.
5. Review the changed cases in the job summary or in the `golden-dataset-reports-*`
   artifact (`comparison.md`, `changed-cases.jsonl`, `junit.xml`).
6. Change the model to `mock-v3-fixed`.
7. Commit and push.
8. Watch the gate pass. The baseline in `baselines/` is untouched throughout.

To make the gate block merges, add the **Golden dataset gate** and
**Prevent moving the goalposts** jobs as required status checks in branch protection.

## Commands

| Command | Purpose |
|---|---|
| `python -m app.cli evaluate --config config/model.yaml --dataset data/golden_dataset.jsonl --output reports/candidate` | Run a configuration; writes `summary.json`, `predictions.jsonl`, `run-manifest.json` |
| `python -m app.cli gate --baseline baselines --candidate reports/candidate --policy config/gates.yaml --output reports/comparison` | Compare and apply gates; exit 1 on hard-gate failure |
| `python -m app.cli approve-baseline [--source reports/candidate]` | Replace the baseline; refuses unless `ALLOW_BASELINE_UPDATE=true` |
| `python -m app.cli validate-dataset --expect-per-category 15` | Schema, label and duplicate-ID validation |
| `python -m app.cli verify-dataset-hash` | Dataset SHA-256 must match the baseline manifest |
| `python -m app.cli check-change-scope --base <sha>` | Fail if the golden dataset and the system changed together |

Make targets: `install`, `test`, `validate`, `baseline`, `evaluate`, `gate`,
`demo-pass`, `demo-fail`, `demo-fixed`, `demo` (all three), `clean`.

## Gates

Hard gates fail the pipeline; warning gates are reported only.

| Gate | Default | Type |
|---|---|---|
| Same golden dataset (hash and case IDs) | identical | hard |
| Category accuracy | ≥ 90% | hard |
| Accuracy drop versus baseline | ≤ 2.0 points | hard |
| Macro-F1 (four categories) | ≥ 88% | hard |
| Macro-F1 drop versus baseline | ≤ 2.0 points | hard |
| `requires_human` accuracy | ≥ 95% | hard |
| Critical safety errors | 0 | hard |
| p95 latency increase | ≤ 25% | warning |
| Cost per 1,000 tickets increase | ≤ 25% | warning |

A **critical safety error** is a security incident routed without human review:
`requires_human = false` when the expected or predicted category is
`SECURITY_INCIDENT`. It is counted separately so it can never hide inside an average.

Also reported: invalid JSON or schema failures, successful-call rate, retries, median
and p95 latency, tokens, per-category precision/recall/F1, regressions, improvements,
new failures by category, confidence changes, errors and timeouts.

## Guardrails against moving the goalposts

- **The baseline is never overwritten automatically.** CI never calls
  `approve-baseline`, and the command refuses to run without
  `ALLOW_BASELINE_UPDATE=true`. Passing the gate does not redefine "correct".
  To promote a candidate deliberately:
  ```bash
  ALLOW_BASELINE_UPDATE=true python -m app.cli approve-baseline --source reports/candidate
  git add baselines && git commit -m "Approve <model> as baseline"
  ```
- **The dataset and the system cannot change in the same pull request.** If
  `data/golden_dataset.jsonl` changes together with the model config, prompt,
  providers, evaluation logic or gate thresholds, CI fails with:
  ```
  The candidate system and its golden dataset changed together.

  Evaluate system changes against the existing golden dataset first.
  Update the golden dataset through a separate reviewed change.
  ```
- **Every run is pinned.** Manifests record the dataset SHA-256, provider, model,
  prompt hash, configuration hash, pricing hash, timestamp and git commit. CI verifies
  the dataset hash against the baseline before evaluating.
- **Two datasets.** Tune prompts on `data/development_dataset.jsonl` (20 cases). The
  golden dataset is a locked evaluation asset.

## Live providers

The same harness runs real models. Providers read credentials only from the
environment; nothing is committed or printed.

| Provider | Config | Environment variables |
|---|---|---|
| Anthropic Messages API | `config/model.anthropic.yaml` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL` |
| OpenAI-compatible (OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM) | `config/model.openai.yaml` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |

Locally:

```bash
cp .env.example .env            # fill in a key; .env is git-ignored
set -a; source .env; set +a
python -m app.cli evaluate --config config/model.anthropic.yaml --output reports/live
```

Edit the `model:` value to a model your account can use, and add its rates to
`config/pricing.yaml` (otherwise cost is reported as `n/a`).

In GitHub:

1. Settings → Secrets and variables → Actions: add `ANTHROPIC_API_KEY` and/or
   `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if needed). The live job runs in a
   `live-evaluation` environment, so you can also scope secrets and required
   reviewers there.
2. Actions → Golden Dataset Gate → Run workflow → mode `live`, choose the config.
3. The first live run reports metrics only. To gate future live runs, approve a live
   baseline from a trusted run into `baselines-live/` and commit it:
   ```bash
   ALLOW_BASELINE_UPDATE=true python -m app.cli approve-baseline \
     --config config/model.anthropic.yaml --baseline-dir baselines-live
   ```

Live models are not perfectly deterministic even at temperature 0, so keep the
regression tolerances in `gates.yaml` realistic for the model you use.

## Reusing the framework

Swap the business task while keeping the trigger, evaluation, comparison and reporting
architecture:

- `data/*.jsonl` and `app/schemas.py`: your cases, labels and output schema.
- `config/prompt.txt`: your instructions.
- `app/safety.py`: your critical-error definition.
- `config/gates.yaml`: your thresholds.
- `app/providers/`: add a provider by implementing `Provider.classify`.

## Repository layout

```
app/            cli, schemas, evaluator, metrics, comparator, safety, hashing,
                pricing, baseline, change_scope, providers/{base,mock,openai_compatible,anthropic}
config/         model.yaml, prompt.txt, gates.yaml, pricing.yaml, live model examples
data/           golden_dataset.jsonl (60, locked), development_dataset.jsonl (20)
baselines/      approved summary.json, predictions.jsonl, manifest.json
reports/        generated, git-ignored
tests/          48 tests
.github/workflows/golden-dataset-gate.yml
```
