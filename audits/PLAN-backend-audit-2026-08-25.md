# Plan: Backend Audit of YOLOMatic `src/` via 360-backend-audit

| Field | Value |
|---|---|
| Objective | Produce a complete, evidence-backed 360-backend-audit report covering all 35,093 lines of `src/`, with zero code changes applied. |
| Status | Draft |
| Version | 1.0 (2026-08-25) |
| Created | 2026-08-25 |

## 1. Objective & Definition of Done

- **Goal:** Audit the entire backend of YOLOMatic (`src/`, 69 Python files, 35,093 LOC) using the 360-backend-audit method, and deliver a written report. The user reviews findings and decides what gets fixed afterwards. This plan produces a **document**, not a diff.
- **Done when:** The file `audits/backend-audit-2026-08-25.md` exists and contains all nine sections of the 360-backend-audit Audit Report Format (Overview, Dead weight, Accuracy, Duplication & structure, Performance, Observability, Risk register, Implementation plan, Handover plan); every one of the nine audit units named in §6 Phase 0 appears in the report with a per-unit verdict; every finding carries a `file:line` citation; and `git status --porcelain src/` shows the same modified/untracked set as at plan start (see §2 Constraints for the recorded baseline).
- **Success measures:**
  - Coverage: 9 of 9 audit units carry an explicit verdict, including verdicts that read "clean — no findings". Measured by counting unit headings in the report.
  - Evidence: 100% of findings cite `path:line`. Measured by grepping the report for findings without a citation; target zero.
  - Proof status: every accuracy finding is labelled `CONFIRMED` (reproduced by reading the executed code path end to end, or by a scratchpad reproduction) or `SUSPECTED` (reasoning stated, not reproduced). Measured by counting unlabelled accuracy findings; target zero.
  - Actionability: the handover section lets a fresh agent start Task 1 without asking a question. Measured by the executor's test in Task 5.4.
- **Must not happen:**
  - No file under `src/`, `tests/`, `configs/`, or `pyproject.toml` is modified, created, or deleted by this plan.
  - No finding is reported without evidence; no invented or speculative defect is presented as fact.
  - No uncommitted work in the tree (the Roboflow upload feature) is reverted, stashed, or committed.
  - No area is silently skipped; a unit that is not audited is reported as not audited, with the reason.

## 2. Context & Constraints

- **Background:** YOLOMatic is a CLI-driven computer-vision training toolkit (YOLO/RF-DETR/SAM/Detectron2) with an interactive terminal interface. The backend has grown to 35k LOC across ten modules with no prior whole-codebase audit — `find` over the repo returns no audit document outside `.venv`. Prior audits were narrow and undocumented in-repo (per session history: `prepare_dataset`, ultralytics integration, smart-balance algorithm). A Roboflow dataset-upload feature is currently uncommitted in the working tree.
- **Constraints:**
  - **Read-only.** The audit reads code and runs the existing test suite; it writes only inside `audits/` and the session scratchpad.
  - **Dirty tree.** At plan start `git status` reports modified `pyproject.toml`, `src/cli/run.py`, `src/cli/upload.py`, and untracked `src/cli/upload_dataset.py`, `src/datasets/validate_roboflow.py`, `src/integrations/roboflow_dataset.py`, `tests/test_roboflow_dataset_uploader.py`, `tests/test_upload_dataset_cli.py`, `tests/test_validate_roboflow.py`. This uncommitted code is **in scope** for the audit and must survive it untouched.
  - **Known test instability.** Session history records a test hang that was isolated but whose root cause is unresolved. The baseline run (Task 0.2) must record it rather than fix it.
  - **Heavy dependencies.** `torch`, `tensorflow`, `tensorrt`, `detectron2`, `rfdetr` — some code paths cannot be executed in this environment. Such paths are audited by reading, and findings from them are marked `SUSPECTED` unless the logic is provably wrong without execution.
  - **Lint scope is narrow.** `[tool.ruff.lint] select = ["E", "F"]` with `E501` ignored — so unused imports (F401) and undefined names are caught, but no dead-code, complexity, or bug-pattern rules run today. Extra analyzers may be run **read-only** in the scratchpad; none are added to `pyproject.toml` by this plan.
  - **Tooling available:** Python 3.12, `pytest` (`testpaths = ["tests"]`, `pythonpath = ["."]`), `ruff>=0.12`, `uv`, `git`, `grep`/`find`.
- **Stakeholders:**
  - *Affected:* users of the `yolomatic`, `yolomatic-train`, `yolomatic-predict`, `yolomatic-upload`, `yolomatic-tensorboard`, `bump` entry points.
  - *Decides:* Shahab (sole maintainer) — approves which findings become work.
  - *Executes:* the AI agent or engineer running this plan; a later agent executes the fixes via the handover section.
  - *Can block:* Shahab; also a red baseline test suite (Checkpoint 0).

## 3. Strategy

- **Chosen path:** Audit in five phases. Phase 0 establishes a factual baseline (behavior map, call graph, test state) so nothing later is guessed. Phases 1–4 run the four 360-backend-audit lenses in the priority the user set — accuracy, then dead weight and duplication, then performance, then observability — each sweeping all nine audit units so a lens is never half-applied. Phase 5 synthesizes findings into the risk register, implementation plan, and handover brief. Findings accumulate in a single scratchpad ledger throughout, and the report is assembled once, at the end, from that ledger.
- **Why it wins:** Lens-major order (one lens across all units) beats unit-major order (all lenses on one unit) for this codebase because the highest-value findings are cross-cutting — the same validation repeated in `cli/` and `datasets/`, the same swallowed-exception pattern in 43 places — and those are only visible when one lens sweeps the whole tree at once. Running accuracy first means correctness findings are in hand before any performance or structural suggestion is written, so no recommendation can be made that trades accuracy away unknowingly. Baselining first (Phase 0) is scheduled earliest because a red or hanging test suite invalidates the "nothing broke" guarantee that the whole audit rests on.
- **Alternatives rejected:**
  - *Unit-major (audit each module fully, then move on):* rejected — cross-module duplication and repeated anti-patterns become invisible, and a per-module report is harder to prioritize.
  - *Automated-tool-first (run linters/complexity tools, report their output):* rejected — `ruff` with `E`+`F` finds almost nothing beyond unused names, and tool output is not evidence of a defect. Tools are used in Phase 2 as *leads*, never as findings.
  - *Audit-and-fix in one pass:* rejected by the user's explicit choice of report-only. It also breaks the "must not happen" constraint of preserving the dirty tree intact.
  - *Sampling (audit the 10 largest files only):* rejected — the user chose all of `src/`; the 14 largest files are 21k of 35k LOC, leaving a third of the backend, including all of `integrations/` and `models/`, unexamined.

## 4. Scope

- **In scope:** Every `.py` file under `src/` — all nine audit units defined in Task 0.1, including the uncommitted Roboflow files. Reading `tests/` is in scope as *evidence* about backend behavior and coverage gaps. `pyproject.toml` is in scope as read-only context (dependencies, entry points, lint/test config).
- **Explicitly out of scope:**
  - Applying any fix, refactor, or deletion — this plan is report-only by the user's decision.
  - Terminal UI/presentation concerns in `src/utils/tui.py` (1,874 LOC) and the interactive menus of `src/cli/run.py` — the 360-backend-audit skill excludes UI and presentation. The *logic* those files invoke (argument construction, config resolution, dispatch, state mutation) remains in scope; screen layout, rendering, colors, and keystroke handling do not.
  - `docs/`, `site/`, `mkdocs.yml`, `README.md`, and the published documentation set.
  - `tests/` as a subject of audit (test quality, test refactoring). Missing tests are *reported* as coverage gaps in Phase 5; existing tests are not critiqued.
  - Dependency vulnerability scanning, license auditing, and CI configuration.
  - Model accuracy, training hyperparameters, and machine-learning methodology — this audits code, not model quality.
- **Change policy:** New requests that arrive mid-audit are **parked** in a "Deferred requests" list at the end of the scratchpad ledger and surfaced in the final report, unless the request invalidates an assumption in §5 — in which case the audit stops at the current checkpoint and the plan is revised before continuing. Requests to *fix* something found during the audit are always parked: fixing is the next plan's job, not this one's.

## 5. Assumptions

| # | Assumption | Validated by |
|---|---|---|
| A1 | The existing test suite passes and is a usable behavioral baseline. Session history claims 278 tests passing but also records an unresolved hang. | Task 0.2 — record pass/fail/hang counts; Checkpoint 0 gates on the result. |
| A2 | The nine audit units in Task 0.1 partition `src/` with no file left unassigned. | Task 0.1 — the unit manifest's file list must equal `find src -name '*.py'` exactly, verified by diffing the two lists. |
| A3 | The six `[project.scripts]` entry points are the complete set of externally reachable entry points into the backend. | Task 0.3 — cross-check against `__main__.py` and any module invoked via `python -m`. |
| A4 | The 388 `print(` call sites and 9 files importing `logging` accurately reflect the observability posture (grep counts taken at plan start). | Task 4.1 — re-derive both counts and classify each `print` site as user-facing CLI output or diagnostic output. |
| A5 | The 163 broad `except` clauses and 43 `except`-then-`pass` sites are real error-swallowing sites rather than grep artifacts (for example, matching a comment or a string). | Task 1.2 and Task 2.3 — inspect every match individually and record the true count. |
| A6 | Code paths gated behind unavailable heavy dependencies (`detectron2`, `tensorrt`, CUDA) can be audited by reading without execution. | Task 0.4 — list which units contain such paths and mark them read-only-audit before Phase 1 begins. |
| A7 | No file under `src/` is generated or vendored (that is, every file is hand-maintained and fair to audit). | Task 0.1 — flag any file with a generation header; excluded units are named in the report. |

## 6. Phases & Tasks

### Phase 0: Baseline & Map

**Checkpoint:** The unit manifest covers 100% of `src/*.py`; the baseline test result is recorded with exact pass/fail/hang counts; the call graph names every entry point; and `git status --porcelain` matches the §2 recorded baseline. If the test suite fails or hangs, record the failing test IDs and continue — but every later finding touching those paths is marked `SUSPECTED`, and Checkpoint 0 is annotated "baseline degraded".

**Task 0.1 — Build the audit unit manifest**
- What: Partition all 69 files of `src/` into the nine audit units, with LOC and a one-line purpose per unit.
- How: (1) Run `find src -name '*.py' -not -path '*__pycache__*' | sort > scratchpad/all_files.txt`. (2) Write the manifest assigning each file to exactly one unit: **U1 cli** (13,549 LOC), **U2 utils** (6,047), **U3 datasets** (3,866), **U4 config** (3,500), **U5 benchmark** (2,757), **U6 augmentation** (2,385), **U7 trainers** (1,394), **U8 models** (975), **U9 integrations** (578). (3) Diff the manifest's file list against `all_files.txt`; resolve any difference. (4) Record for each file whether it is tracked, modified, or untracked in git.
- Where: `scratchpad/unit-manifest.md`.
- Depends on: none | Parallel: yes (with 0.2, 0.3)
- Effort: S | Priority: must
- Done when: `diff` between the manifest file list and `all_files.txt` is empty, and every unit has a stated LOC and purpose.

**Task 0.2 — Record the test baseline**
- What: Capture the exact current state of the test suite as the behavioral baseline the audit must not disturb.
- How: Run `uv run pytest -q --timeout=300 2>&1 | tee scratchpad/baseline-tests.txt` (if `pytest-timeout` is unavailable, run without `--timeout` under a wall-clock limit of 20 minutes and record any hang instead of waiting). Record: total collected, passed, failed, skipped, errors, wall time, and the node IDs of anything failing or hanging. Do not fix anything.
- Where: `scratchpad/baseline-tests.txt` plus a summary line in the ledger.
- Depends on: none | Parallel: yes
- Effort: S | Priority: must
- Done when: `scratchpad/baseline-tests.txt` exists and the ledger records the five counts plus any failing/hanging node IDs, or an explicit "suite did not terminate within 20 minutes" note.

**Task 0.3 — Map entry points and call graph**
- What: Document how execution enters the backend and how control reaches each unit, so dead-code claims later have a reachability basis.
- How: (1) List the six `[project.scripts]` entry points (`bump`, `yolomatic`, `yolomatic-predict`, `yolomatic-tensorboard`, `yolomatic-train`, `yolomatic-upload`) and `src/__main__.py`. (2) For each, trace the import and dispatch chain to the modules it reaches, using `grep -rn "^from src\.\|^import src\." src` to build the import edges. (3) Produce a per-unit list of inbound callers. (4) Mark any module with zero inbound edges from any entry point as a **reachability lead** (not yet a finding — dynamic imports and plugin lookups must be ruled out in Task 2.1).
- Where: `scratchpad/call-graph.md`.
- Depends on: 0.1 | Parallel: no
- Effort: M | Priority: must
- Done when: every unit has a named inbound caller or is listed as a reachability lead, and every one of the six entry points has its dispatch chain written down.

**Task 0.4 — Identify unexecutable paths and trust boundaries**
- What: Record which code cannot be run in this environment, and where external/untrusted data enters the backend — the two facts that set proof standards for Phase 1.
- How: (1) Grep for optional/heavy imports (`detectron2`, `tensorrt`, `tensorflow`, `coremltools`, `ncnn`, `openvino`, CUDA guards) and for the `src/utils/ml_dependencies.py` gating logic; list the guarded paths per unit. (2) List trust boundaries: filesystem reads of user-supplied datasets and label files, YAML/JSON/NDJSON parsing, environment variables and `.env`, subprocess invocations, and every outbound network call (Roboflow, Ultralytics HUB, ClearML, HuggingFace).
- Where: `scratchpad/boundaries.md`.
- Depends on: 0.1 | Parallel: yes (with 0.3)
- Effort: M | Priority: must
- Done when: every unit is marked executable or read-only-audit, and each trust boundary is listed with its `file:line` and the data format crossing it.

**Task 0.5 — Open the findings ledger**
- What: Create the single accumulating record every later task writes into, so the report can be assembled once at the end without re-reading code.
- How: Create a table with columns: ID (`ACC-01`, `DEAD-01`, `DUP-01`, `PERF-01`, `OBS-01`), unit, `file:line`, lens, description, evidence, proof status (`CONFIRMED`/`SUSPECTED`/`CLEAN`), severity (`critical`/`high`/`medium`/`low`), and suggested action. Append a "Deferred requests" section at the bottom per the §4 change policy.
- Where: `scratchpad/findings-ledger.md`.
- Depends on: none | Parallel: yes
- Effort: S | Priority: must
- Done when: the ledger file exists with all nine columns and the deferred section.

### Phase 1: Accuracy & Correctness (highest priority)

**Checkpoint:** All nine units carry an accuracy verdict in the ledger; every accuracy finding is labelled `CONFIRMED` or `SUSPECTED` with its reasoning written out; units with no findings are recorded as `CLEAN`. No fix has been applied.

**Task 1.1 — Audit business logic and boundary conditions**
- What: Find off-by-one errors, inverted conditions, wrong operators, incorrect defaults, and mishandled empty/single-element cases across all nine units.
- How: Read each unit's core logic with attention to: dataset split ratios and index arithmetic in `src/datasets/prepare.py` and `src/datasets/core.py`; class-balancing and rare-class arithmetic; augmentation variant counts and bounding-box/keypoint/polygon coordinate transforms in `src/augmentation/engine.py` and `transforms.py`; metric computation in `src/benchmark/metrics.py`; config value resolution and defaulting in `src/config/parameters.py` and `generator.py`. For each suspected defect, trace the executed path end to end; where the logic is pure and importable, write a throwaway reproduction under the scratchpad to confirm it. Label confirmed only when reproduced or provable by reading; otherwise `SUSPECTED` with the reasoning stated.
- Where: all nine units; findings to the ledger as `ACC-*`.
- Depends on: 0.3, 0.4, 0.5 | Parallel: yes (with 1.2, 1.3, 1.4)
- Effort: L | Priority: must
- Done when: every unit has a logic verdict in the ledger, and each `ACC-*` finding has a proof status plus reasoning.

**Task 1.2 — Audit failure semantics and error paths**
- What: Determine where failures are silently swallowed, misreported as success, or leave state partially written.
- How: Enumerate every broad handler with `grep -rn "except Exception\|except BaseException\|except:" src --include=*.py` (163 matches at plan start) and every `except`-then-`pass` (43 at plan start). Inspect each site individually and classify as: (a) legitimate — narrow, intentional, with a comment or an alternative path; (b) swallowing — failure is hidden and the caller cannot detect it; (c) masking — an error is caught and a success-shaped value returned. Separately, check that multi-step filesystem writes (dataset staging, augmentation output, archive/extract, config generation) either complete fully or leave no partial artifact, and that partial success is never reported as success to the CLI caller.
- Where: all nine units; findings to the ledger as `ACC-*` with sub-tag `error-path`.
- Depends on: 0.5 | Parallel: yes
- Effort: L | Priority: must
- Done when: the true counts of categories (a)/(b)/(c) are recorded, every (b) and (c) site is a ledger entry with `file:line`, and A5 is marked validated or corrected.

**Task 1.3 — Audit data integrity and concurrency**
- What: Find race conditions, duplicate processing, non-atomic writes, and unsafe ordering assumptions in the parallel and long-running paths.
- How: Read every use of `ThreadPoolExecutor`, `ProcessPoolExecutor`, `multiprocessing`, `fwalk`, threading, and shared mutable state — the augmentation worker pool in `src/augmentation/engine.py`, parallel dataset listing in `src/datasets/`, benchmark thumbnail generation in `src/benchmark/thumbnails.py`, and the `--type` threading in the Roboflow upload path. For each: identify shared state, check whether writes are atomic (temp-file-then-rename versus in-place), check whether a retried or re-run job double-processes, and check whether ordering is assumed where the executor gives none. Also verify that dataset split assignment is deterministic and leak-free given the group-split logic added in commit `b0d1d30`.
- Where: U3 datasets, U6 augmentation, U5 benchmark, U9 integrations, U1 cli; findings to the ledger as `ACC-*` with sub-tag `concurrency`.
- Depends on: 0.4, 0.5 | Parallel: yes
- Effort: L | Priority: must
- Done when: every concurrent or shared-state site from the grep list has a written verdict, and each atomicity risk names the specific artifact that could be left partial.

**Task 1.4 — Audit trust boundaries and input validation**
- What: Confirm that every input crossing a trust boundary is validated before use, and that nothing upstream is trusted blindly.
- How: For each boundary listed in `scratchpad/boundaries.md`: trace the value from entry to first use; record whether type, range, existence, and shape are checked; and note what happens on malformed input (clear error versus crash versus silent wrong result). Cover at minimum: label file parsing (YOLO txt, COCO JSON, NDJSON), dataset YAML, `.env` and environment variables, API responses from Roboflow/Ultralytics/ClearML, user-supplied paths (including traversal outside the intended root), and subprocess argument construction.
- Where: U3 datasets, U4 config, U9 integrations, U2 utils, U1 cli; findings to the ledger as `ACC-*` with sub-tag `trust-boundary`.
- Depends on: 0.4, 0.5 | Parallel: yes
- Effort: M | Priority: must
- Done when: every boundary in `boundaries.md` has a validation verdict and a stated malformed-input behavior.

**Task 1.5 — Audit numeric and temporal accuracy**
- What: Find precision loss, rounding errors, unit confusion, and timezone/calendar defects.
- How: Review float-to-int conversions in coordinate and box math, normalization/denormalization round-trips in `src/augmentation/transforms.py` and the dataset format converters, percentage and ratio arithmetic in split and balance logic, metric aggregation and averaging in `src/benchmark/metrics.py`, and every use of `datetime.now()`, `utcnow()`, `time.time()`, and timestamp formatting for naive/aware mixing.
- Where: U3, U5, U6, U2; findings to the ledger as `ACC-*` with sub-tag `numeric`.
- Depends on: 0.5 | Parallel: yes
- Effort: M | Priority: should
- Done when: every coordinate round-trip and every `datetime`/`time` call site has a verdict recorded.

### Phase 2: Dead Weight & Duplication

**Checkpoint:** Every dead-weight claim carries reachability evidence; every duplication claim names two or more concrete `file:line` sites and states whether the logic is genuinely identical in purpose; no removal is proposed without evidence.

**Task 2.1 — Find dead code**
- What: Identify unreachable branches, unused functions, unused parameters, orphaned modules, commented-out blocks, and stale config.
- How: (1) Run read-only analyzers in the scratchpad as lead generators only — `uv run ruff check --select F401,F811,F841,ARG,ERA --no-fix src` and, if installable without touching `pyproject.toml`, `vulture src` — writing output to the scratchpad. (2) For every lead, verify by hand: grep the symbol name across `src/`, `tests/`, `configs/`, and `pyproject.toml`, and check for dynamic access (`getattr`, `globals()`, string-keyed dispatch tables, entry points, plugin registries) before calling it dead. (3) Cross-reference the reachability leads from Task 0.3. (4) Check `src/config/` for config keys no code reads, and the parameter catalogs in `src/config/parameters.py` for entries no longer referenced.
- Where: all nine units; findings to the ledger as `DEAD-*`.
- Depends on: 0.3, 0.5 | Parallel: yes (with 2.2, 2.3)
- Effort: L | Priority: must
- Done when: every `DEAD-*` entry states the grep evidence and the dynamic-access check result; leads that failed verification are recorded as "not dead — reason".

**Task 2.2 — Find duplicated logic**
- What: Locate the same computation, validation, query, or transformation implemented more than once, and judge whether unification is correct.
- How: Search for structural repetition across the CLI/domain seam in particular — path resolution and dataset-root discovery, YAML read/write helpers, split-ratio validation, class-name normalization, image-extension filtering, progress reporting, and API-client construction. Use `grep` for repeated literal sets (image extensions, split names, format names) and repeated helper shapes. For each candidate, state whether the two sites serve the *same purpose* (unify) or merely *look alike* (leave separate — premature abstraction is debt). Note that `src/cli/` is 13.5k LOC against `src/datasets/` 3.9k; check specifically whether `cli/` holds domain logic that belongs in a domain unit.
- Where: primarily U1 cli versus U3 datasets, U4 config, U9 integrations; findings to the ledger as `DUP-*`.
- Depends on: 0.5 | Parallel: yes
- Effort: L | Priority: must
- Done when: every `DUP-*` entry lists two or more `file:line` sites and carries an explicit unify/leave-separate judgment with its reason.

**Task 2.3 — Find incomplete logic**
- What: Identify placeholder returns, partial validation, empty handlers, and abandoned work.
- How: Grep for `TODO|FIXME|XXX|HACK` (0 matches at plan start — record this as a clean result), `NotImplementedError`, `pass  #`, bare `return None` on error paths, and functions whose docstring promises behavior the body does not implement. Reuse the category (b)/(c) sites from Task 1.2 rather than re-deriving them. Check for stale feature flags and dead branches in `src/config/settings.py`.
- Where: all nine units; findings to the ledger as `DEAD-*` with sub-tag `incomplete`.
- Depends on: 1.2, 0.5 | Parallel: no
- Effort: M | Priority: must
- Done when: each grep pattern has a recorded count and every non-zero match has a verdict; the zero-`TODO` result is stated as a clean finding.

**Task 2.4 — Assess module structure and boundaries**
- What: Judge whether responsibilities sit in the right module, and identify the specific splits worth making.
- How: For each of the seven files over 1,000 LOC (`cli/run.py` 4,095; `utils/tui.py` 1,874; `config/parameters.py` 1,868; `cli/augment.py` 1,522; `augmentation/engine.py` 1,438; `datasets/prepare.py` 1,437; `config/generator.py` 1,373; `utils/ai_client.py` 1,297; `benchmark/report.py` 1,075; `cli/benchmark.py` 1,044; `cli/upload.py` 1,040; `cli/upload_dataset.py` 1,034; `datasets/core.py` 1,021), list the distinct responsibilities it holds and whether transport (CLI/TUI), domain logic, and data access are separated. Propose a concrete split only where a responsibility has a clean seam; state explicitly where a large file is cohesive and should stay whole. Also evaluate hand-rolled helpers (retry loops, validation, serialization, pagination) against mature libraries, and for each library suggestion verify maintenance status, adoption, license compatibility, and that it is lighter than the code it replaces — reject any that fails one of the four.
- Where: the thirteen files listed above, plus each unit's `__init__.py`; findings to the ledger as `DUP-*` with sub-tag `structure`.
- Depends on: 2.2 | Parallel: no
- Effort: L | Priority: should
- Done when: every file over 1,000 LOC has a split-or-keep verdict with reasoning, and every library recommendation records all four verification results.

### Phase 3: Performance

**Checkpoint:** Every performance opportunity is ranked by impact against effort, carries a measurement plan, and is explicitly checked against the accuracy findings from Phase 1 so that no suggestion trades correctness for speed.

**Task 3.1 — Establish performance baselines**
- What: Measure current cost on the paths that matter, so later claims are measurable rather than asserted.
- How: Identify the three to five hottest realistic workloads from the entry-point map — dataset preparation over a large image set, augmentation over a large set, dataset listing/scanning, benchmark evaluation. For each, either time an existing run using data already present under `datasets/` or `output/`, or record why measurement is not possible in this environment (dataset absent, GPU required) and mark any related opportunity `UNMEASURED`. Use `cProfile` or wall-clock timing written to the scratchpad; run nothing that writes into `src/`.
- Where: `scratchpad/perf-baseline.md`.
- Depends on: 0.2 | Parallel: no
- Effort: M | Priority: should
- Done when: each selected workload has a recorded baseline number or an explicit "not measurable here — reason".

**Task 3.2 — Audit I/O and data-access patterns**
- What: Find repeated filesystem traversal, unbounded scans, over-reading, and blocking calls that could run concurrently.
- How: Trace every directory walk (`os.walk`, `fwalk`, `Path.glob`, `rglob`) and count how many times the same tree is traversed per command; check whether image dimensions or label contents are read more than once per file; check for full-image decode where only metadata is needed; check whether network calls to Roboflow/Ultralytics/HuggingFace are batched, retried with backoff, and paginated; check whether cache layers in `src/datasets/cache.py` are hit on the paths that would benefit.
- Where: U3, U6, U5, U9, U2; findings to the ledger as `PERF-*` with sub-tag `io`.
- Depends on: 0.3, 0.5 | Parallel: yes (with 3.3)
- Effort: M | Priority: should
- Done when: every traversal site is counted per command path, and each `PERF-*` entry states expected impact, effort, and the measurement that would prove it.

**Task 3.3 — Audit computation and algorithmic cost**
- What: Find repeated work, wrong data structures, and complexity that matters at real dataset scale.
- How: Look for linear scans inside loops over images or labels (quadratic behavior at scale), list membership tests that should be set tests, recomputation of values stable across a run (config resolution, class maps, signatures), and per-image work that could be hoisted. Weigh each against realistic scale — the session record cites an 86.6k-image, 7.0 GB dataset; use that as the scale target.
- Where: U3, U6, U5; findings to the ledger as `PERF-*` with sub-tag `compute`.
- Depends on: 0.5 | Parallel: yes
- Effort: M | Priority: should
- Done when: each `PERF-*` entry states current complexity, proposed complexity, and the input size at which the difference becomes material.

**Task 3.4 — Flag performance trade-offs explicitly**
- What: Separate free wins from wins that cost something, so no risky speedup hides inside a recommendation.
- How: Review every `PERF-*` entry and tag it `FREE` (no behavioral change) or `TRADE-OFF` (names what is given up — cache staleness, float reordering, reduced validation, added concurrency hazard, higher memory). Cross-check each against the Phase 1 accuracy findings; any suggestion touching a path with an open `ACC-*` finding is marked "blocked until ACC-xx resolved".
- Where: the ledger.
- Depends on: 1.1, 1.3, 3.2, 3.3 | Parallel: no
- Effort: S | Priority: must
- Done when: every `PERF-*` entry carries a `FREE`/`TRADE-OFF` tag and a cross-check result against the accuracy findings.

### Phase 4: Observability

**Checkpoint:** The observability gap analysis covers all nine units, and every proposed instrumentation change is additive — wrapping existing logic, never restructuring it.

**Task 4.1 — Audit the current logging and output posture**
- What: Establish what the backend actually emits today and whether a production failure could be diagnosed from it.
- How: Re-derive the counts (388 `print(` sites, 9 files importing `logging` at plan start) and classify every `print` site as *user-facing CLI output* (legitimate — this is a CLI tool) or *diagnostic output* (should be a logged event). For the 9 logging-aware files, record whether levels are used correctly and whether a logger is configured at all. Check whether any output path can emit secrets — API keys from `.env`, Roboflow/Ultralytics/HuggingFace tokens, absolute paths revealing user identity.
- Where: all nine units; findings to the ledger as `OBS-*`.
- Depends on: 0.5 | Parallel: yes (with 4.2)
- Effort: M | Priority: must
- Done when: every `print` site is classified, the secret-leak check is recorded with its result, and A4 is marked validated or corrected.

**Task 4.2 — Audit error context and traceability**
- What: Determine whether a failed run can be diagnosed after the fact.
- How: For each swallowing/masking site from Task 1.2, record what context is lost (input values, file path, stage). Check whether long-running jobs (training, augmentation, dataset prep, upload) emit stage transitions, progress, and a terminal success/failure record; whether a run has a correlation identifier tying its artifacts, logs, and config together; and whether retries are visible when they occur.
- Where: U1, U3, U6, U7, U9; findings to the ledger as `OBS-*` with sub-tag `traceability`.
- Depends on: 1.2, 0.5 | Parallel: no
- Effort: M | Priority: must
- Done when: each long-running job path has a traceability verdict, and each lost-context site names the specific values that should be captured.

**Task 4.3 — Write the non-breaking instrumentation plan**
- What: Turn the observability gaps into a change plan that cannot alter behavior.
- How: For each `OBS-*` gap, specify the additive change — wrap the call, add a logger call adjacent to existing logic, introduce a decorator — and state explicitly what is *not* being restructured. Specify a single logging configuration point and how CLI user output stays separate from diagnostic logging so terminal UX does not regress. Mark any insertion inside a per-image or per-item loop as a hot path and either exclude it or gate it behind a debug level. Stage anything riskier than a log line behind a config flag.
- Where: the ledger and report section 6.
- Depends on: 4.1, 4.2 | Parallel: no
- Effort: M | Priority: must
- Done when: every `OBS-*` gap has an additive remedy, hot-path insertions are marked and gated, and no remedy requires rewriting existing logic to accommodate logging.

### Phase 5: Synthesis & Delivery

**Checkpoint:** The report contains all nine 360-backend-audit sections, passes the executor's test, and the working tree is verified unchanged.

**Task 5.1 — Build the risk register**
- What: State what could break when the findings are eventually acted on, and how each risk is detected.
- How: For every finding proposed for change, record: what breaks if the change is wrong, how it would be detected (which test, which observable behavior), and the mitigation or rollback. Include the risks of *not* acting on the critical accuracy findings. Include the baseline test instability from Task 0.2 as a standing risk.
- Where: report section 7.
- Depends on: 1.1–1.5, 2.1–2.4, 3.4, 4.3 | Parallel: no
- Effort: M | Priority: must
- Done when: every proposed change has a matching risk row with a detection method and a mitigation.

**Task 5.2 — Classify findings and order the implementation plan**
- What: Convert the ledger into a safest-first execution sequence the user can approve in batches.
- How: Classify every finding as **safe to apply now** (no behavior change, covered by existing tests), **needs tests first** (behavior-affecting, coverage gap present — name the missing boundary/failure/concurrency test), or **needs human decision** (trade-off, ambiguous intent, or architectural). Then order: confirmed critical accuracy fixes → confirmed non-critical accuracy fixes → evidence-backed dead code removal → observability additions → duplication unification → performance changes → structural refactors. State why this order is safest (each earlier step reduces the risk of the next).
- Where: report section 8.
- Depends on: 5.1 | Parallel: no
- Effort: M | Priority: must
- Done when: every ledger finding carries one of the three classifications and one position in the ordered plan; findings needing tests name the specific missing test.

**Task 5.3 — Assemble the audit report**
- What: Write the deliverable.
- How: Create `audits/backend-audit-2026-08-25.md` with exactly the nine 360-backend-audit sections. Section 1 states scope and per-unit health. Sections 2–6 present findings from the ledger grouped by lens, each with `file:line` and evidence, and each clean unit stated as clean. Section 9 is the handover brief for the next agent: context, decisions (what is deliberately *not* being changed and why), tasks in order with exact locations, per-task verification, risks, and current state (all findings pending — nothing applied). Do not add the file to `mkdocs.yml` nav; this is an internal document, not published documentation.
- Where: `audits/backend-audit-2026-08-25.md`.
- Depends on: 5.1, 5.2 | Parallel: no
- Effort: L | Priority: must
- Done when: all nine sections exist, every unit appears with a verdict, and a grep for findings lacking a `file:line` citation returns zero.

**Task 5.4 — Verify the audit against its own quality gate**
- What: Confirm the audit meets the 360-backend-audit quality gate before delivery.
- How: Walk the skill's eleven quality-gate questions against the report and answer each in writing. Run the executor's test: re-read handover section 9 as if with no prior context and list every point where a question would arise; sharpen until the list is empty. Re-run `uv run pytest -q` and confirm the result matches the Task 0.2 baseline exactly. Run `git status --porcelain` and confirm it matches the §2 recorded baseline.
- Where: report appendix plus terminal verification.
- Depends on: 5.3 | Parallel: no
- Effort: M | Priority: must
- Done when: all eleven gate questions are answered yes in writing, the executor's-test question list is empty, the test result matches baseline, and `git status --porcelain` matches the §2 baseline.

**Task 5.5 — Deliver and offer the handover format**
- What: Hand the report over and ask how the handover should be issued.
- How: Summarize the top findings by severity in the terminal. Ask the user whether the handover brief should be issued as a standalone prompt for the next agent or kept as report section 9 (the 360-backend-audit skill requires this question be asked). Offer to publish the report as an Artifact.
- Where: terminal.
- Depends on: 5.4 | Parallel: no
- Effort: S | Priority: must
- Done when: the summary is delivered and the handover-format question is asked.

## 7. Risks & Countermeasures

| Risk | Impact | Countermeasure |
|---|---|---|
| The baseline test suite hangs (known unresolved issue) | The "nothing broke" guarantee has no foundation; every finding's confidence drops | Task 0.2 records the hang with a 20-minute wall limit and continues; Checkpoint 0 annotates "baseline degraded" and findings on affected paths are marked `SUSPECTED` |
| Findings are reported without proof because heavy dependencies cannot run here | The report asserts bugs that do not exist, destroying its credibility | Task 0.4 marks unexecutable units up front; the `CONFIRMED`/`SUSPECTED` label is mandatory on every accuracy finding and is checked in Task 5.4 |
| Audit fatigue at 35k LOC causes silent skipping of later units | Coverage is claimed but not delivered | The unit manifest (Task 0.1) forces a per-unit verdict in every phase checkpoint; a unit with no verdict blocks the checkpoint |
| The uncommitted Roboflow work is accidentally modified or lost | The user loses in-progress work | Read-only constraint in §2; `git status --porcelain` compared against the recorded baseline in Task 5.4 |
| Duplication is unified where the logic only looks alike | A future refactor couples unrelated behaviors and breaks one of them | Task 2.2 requires an explicit same-purpose judgment with reasoning per candidate; look-alikes are recorded as leave-separate |
| A performance recommendation silently costs accuracy | A fast wrong answer ships | Task 3.4 tags every entry `FREE` or `TRADE-OFF` and blocks any entry touching an open accuracy finding |
| Dead code is proposed for removal but is reached dynamically | A future removal breaks a working command | Task 2.1 requires a dynamic-access check (`getattr`, `globals()`, dispatch tables, entry points) before any dead label |
| A library recommendation is heavier or less maintained than the code it replaces | Dependency burden grows for no gain — a real risk given the already-large dependency set in `pyproject.toml` | Task 2.4 requires all four checks (maintained, adopted, license-compatible, lighter) recorded per recommendation; failing one rejects it |
| Observability suggestions land inside per-image loops | Instrumenting the fix degrades the performance the audit set out to protect | Task 4.3 marks hot-path insertions and either excludes them or gates them behind a debug level |
| Scope creeps into fixing things found mid-audit | The report is never finished and the tree is no longer clean | §4 change policy parks all fix requests in the ledger's deferred section |

## 8. Verification & Replanning

- **Per task:** each task's "Done when" is the acceptance check; a task is not complete until that check is observed.
- **Per phase:** the phase Checkpoint must hold before the next phase begins. A failed checkpoint means finishing the phase, not proceeding with a gap.
- **Overall:** the plan is complete when (1) `audits/backend-audit-2026-08-25.md` contains all nine sections with a verdict for all nine units, (2) Task 5.4's eleven quality-gate answers are all yes in writing, (3) `uv run pytest -q` matches the Task 0.2 baseline, and (4) `git status --porcelain` matches the §2 recorded baseline.
- **Replan when:**
  - Assumption A2 fails — files exist in `src/` that no unit covers (the manifest and effort estimates are wrong).
  - Assumption A1 fails badly — the suite is broadly red rather than a single known hang (baseline is unusable; fix or explicitly accept degraded confidence before Phase 1).
  - A critical accuracy finding implies user-visible data corruption in shipped datasets — stop and surface it immediately rather than waiting for Phase 5.
  - The user changes the deliverable from report-only to apply-fixes (this plan's every "must not happen" changes).
  - Phase 1 alone produces more than roughly 50 findings — re-scope the remaining phases with the user rather than diluting depth across all four lenses.

## 9. Traceability

| Objective | Covered by tasks |
|---|---|
| Complete coverage of all 35,093 LOC in `src/` | 0.1, 0.3, and the per-unit verdict requirement in 1.1, 1.2, 2.1, 2.2, 4.1 |
| Accuracy and correctness (user priority 1) | 1.1, 1.2, 1.3, 1.4, 1.5 |
| Dead weight and duplication (user priority 2) | 2.1, 2.2, 2.3, 2.4 |
| Performance (user priority 3) | 3.1, 3.2, 3.3, 3.4 |
| Observability (user priority 4) | 4.1, 4.2, 4.3 |
| Every finding backed by evidence | 0.5 (ledger schema), 2.1 (removal evidence), 5.3, 5.4 |
| No code changed; working tree preserved | §2 read-only constraint, 5.4 git verification |
| Report contains all nine 360-backend-audit sections | 5.1, 5.2, 5.3 |
| Handover lets a fresh agent act with zero guessing | 5.3 (section 9), 5.4 (executor's test), 5.5 (format question) |
| No recommendation trades away correctness | 3.4, 4.3, 5.1 |

## 10. Handover Summary

- **Executor must know:**
  - This plan produces a **document only**. Do not edit, delete, or create any file under `src/`, `tests/`, `configs/`, or `pyproject.toml`. Write only into `audits/` and the session scratchpad.
  - The working tree is intentionally dirty — an uncommitted Roboflow upload feature (see §2 for the exact file list). It is **in scope to audit** and **must not be touched**. Verify with `git status --porcelain` before starting and again at the end.
  - Repository: `/home/shahab/Documents/Hobby/YOLOMatic`, Python 3.12, dependencies managed by `uv`. Tests: `uv run pytest -q` (`testpaths = ["tests"]`, `pythonpath = ["."]`). Lint: `uv run ruff check src` (only `E`+`F` selected, `E501` ignored).
  - Backend layout and sizes: `cli/` 13,549 · `utils/` 6,047 · `datasets/` 3,866 · `config/` 3,500 · `benchmark/` 2,757 · `augmentation/` 2,385 · `trainers/` 1,394 · `models/` 975 · `integrations/` 578 LOC.
  - Signal counts taken at plan start, to be re-derived and verified, not trusted: 163 broad `except` clauses, 43 `except`-then-`pass`, 388 `print(` sites, 9 files importing `logging`, 0 `TODO`/`FIXME` markers.
  - UI and presentation are out of scope; the logic behind the TUI is in scope. See §4 for the exact line.
  - Order matters: Phase 0 before everything; accuracy (Phase 1) before performance (Phase 3), because performance suggestions are cross-checked against accuracy findings in Task 3.4.
  - Every accuracy finding needs a `CONFIRMED` or `SUSPECTED` label. Never present an unproven defect as fact.
  - Findings accumulate in `scratchpad/findings-ledger.md` throughout; the report is assembled once, in Task 5.3.
- **Cut line** (dropped first, in this order, if time or context runs short):
  1. Task 2.4 (module structure and library evaluation) — the largest effort with the least immediate safety value.
  2. Task 1.5 (numeric and temporal accuracy) — narrower blast radius than Tasks 1.1–1.4.
  3. Task 3.1 (performance baselines) — dropping it means every `PERF-*` entry is marked `UNMEASURED` and no performance claim may assert a magnitude.
  4. Tasks 3.2 and 3.3 (performance audit) — drop together; Phase 3 then reports "not audited — cut for scope" rather than partial coverage.
  Never cut: Phase 0, Tasks 1.1–1.4, 2.1–2.3, 4.1–4.3, or any of Phase 5.
- **Open questions:**
  - Should the handover brief be issued as a standalone prompt for the next agent, or kept as section 9 of the report? (Asked in Task 5.5; the 360-backend-audit skill requires the question.)
  - Is the Roboflow upload feature intended to be committed soon? If yes, its accuracy findings should be prioritized ahead of the rest in Task 5.2's ordering.
  - Is there a representative dataset available locally for the Task 3.1 performance baselines, or should all `PERF-*` entries be `UNMEASURED`?
- **Decision points:**
  - **After Checkpoint 0:** if the test baseline is broadly red, decide whether to stabilize it first or proceed with explicitly degraded confidence.
  - **After Phase 1:** if a critical, confirmed accuracy defect implies corrupted output in already-produced datasets, decide whether to halt the audit and fix immediately (overriding the report-only constraint) or continue and report.
  - **After Phase 2:** if the finding count exceeds roughly 50, decide whether to narrow Phases 3–4 or extend the timeline.
  - **At Task 5.5:** decide which findings become the next work batch, and in what order.
