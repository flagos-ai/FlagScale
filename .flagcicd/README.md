# FlagScale test policy and execution contract

`.flagcicd/project.yaml` is the only manually edited repository test-policy file.
Platform images, runner labels, runtime installation, devices, process counts,
include/exclude rules and model parameters remain in their existing platform
and case YAML. Runtime JSON/XML files are generated evidence, not policy.

## Profiles

- `pr`: select affected platforms, unit files and functional cases.
- `branch` and `nightly`: run every enabled configured unit/functional case.
- `requested`: run the configured scope of explicitly requested platforms.

There is no rollout switch or repository variable required to enable routing.
Unknown/shared code and unavailable diffs conservatively use configured full
regression. An unknown test is different: without an ownership/execution rule,
planning fails even if another changed file requests a full regression.

## Test ownership and source impact

`test_groups` contains test-path ownership, source ownership, explicit affected
unit groups, functional selectors and recorded E2E gaps. New ordinary test files
under an existing test directory are discovered automatically. Only helpers,
placeholders and new ownership/execution boundaries need policy edits.

- A test path has exactly one group, or one explicit helper/placeholder exception.
- More specific source paths override parent paths; equal-specificity conflicts
  fail validation. Directory rules end with `/`; other rules are relative globs.
- `affects` means additional tests required when THIS group's source changes.
  It is one-hop impact, not a recursive module dependency graph: selecting a
  consumer's tests does not pretend that consumer's source changed too.
- Functional selectors match configured `task`, and optionally `backend`,
  `model`, or `case`. No case absent from platform support is invented.
- Functional config consumers are indexed from platform case support and local
  Hydra defaults. Shared model configuration conservatively expands consumers.
- Platform labels add platforms; explicitly requested platforms run their full
  configured workload. Ordinary labels do not affect selection.
- `examples/` Python/YAML are runtime inputs, not documentation. Markdown/RST
  and `docs/` remain documentation inputs.

The generated catalog distinguishes `ci-control` from the existing accelerator
unit suite. Mixed-environment unit files are not moved into a CPU job. The
legacy torchrun process count, platform exclusions and launch command are kept.
A new test excluded on every platform fails planning; no full fallback can make
an unwired test appear validated.

## Plan and evidence

`flagcicd_plan.py plan` generates a `flagcicd.plan/v2` document containing source,
control and base SHAs, changed paths, configuration identity (including selected
case/default YAML), selection reasons/gaps, expected unit files/ranks and cases,
gate requirements and a canonical content hash. `slice` validates checkout and
configuration identity before producing existing workflow inputs.

Unit workers write one JUnit and collection manifest per rank. The runner writes
`result.json` with the exact plan-bound execution specification and report
hashes. Collection manifests record actual pytest targets, files, node IDs and
collection-time skips. Functional jobs use the original CLI and numerical
comparator; the comparator JUnit records its actual platform/device/task/model/
case arguments. A wrapper records command outcome and bound functional evidence.

Platform and top-level gates download `ci-evidence-*` artifacts and require:

- Every planned execution identity, with no unexpected or duplicate result.
- Matching plan hash, source SHA, platform/device and current workflow run.
- Valid rank topology and target/file collection for units.
- Valid JUnit, consistent testcase counts, no failures/errors and a passing test.
- Matching functional comparator identity; successful numerical comparison.
- Unchanged report/collection digests after result recording.

Missing or all-skipped evidence fails even if the job status is green. A later
failed attempt cannot use an earlier passing result. Successful jobs from older
attempts of the same run can satisfy a failed-jobs rerun only with identical
source/plan identity. Evidence is execution accounting, not a signed attestation
or a proof of complete semantic source dependency coverage.

## Entrypoints and compatibility

Automatic PR, branch, Nightly and `ci-request.yml` use the same planner,
`_platform_pipeline.yml`, dependency producer, runners and evidence gates.
Requested planning uses controller code with the exact candidate checkout read
as data; test jobs keep read-only permissions and receive no report secrets.
Requested results remain supplemental to `CI Complete`. PR source is rechecked
before completing the requested workflow. Comment commands become active only
when their workflow/controller reaches the default branch.

Legacy manual platform wrappers, image validation and dependency-triggered
callers may still omit `test_plan`. Empty plan/selection inputs explicitly mean
**legacy full diagnostic execution**, not a planned/evidence-gated run. Those
callers are compatibility consumers; their job success must not be described as
passing the new plan/evidence contract. No automatic PR/Nightly/request entry
uses this exception.

Coverage generation/upload and numerical comparators are unchanged. A partial
PR's coverage is not a full-regression coverage measurement. Kunlunxin benchmark
keeps the existing relaxed provisional baseline unchanged, as requested. Its
metadata and tolerance remain intact. A green run against this provisional
baseline does not establish a performance fix; performance is reviewed separately.

## Validation and capability boundaries

```bash
python .github/scripts/ci/flagcicd_contract.py
python -m pytest --confcutdir=tests/ci tests/ci -q
python .github/scripts/ci/flagcicd_plan.py plan --profile nightly --file /tmp/nightly-plan.json
```

Local inventory discovery includes nonignored new tests without requiring git
staging. CI checks the committed checkout. A deleted/renamed test uses both old
and new diff paths; exceptions and execution support must remain consistent.

L0–L4 were capability descriptions, not implemented level-selection interfaces.
They are removed from executable policy rather than advertising nonexistent
level inheritance or release gates. Existing tests still include static/control,
single-process, distributed and benchmark validation. Complete package-install,
universal device smoke, inference/serve E2E, convergence, endurance and release
acceptance remain gaps. No group matrix, scheduler, resource lock or retry-policy
change is introduced by this refactor. Real accelerator and GitHub artifact /
requested-workflow integration still require remote verification.

## CI summary and trusted controllers

The plan summary starts with stage, selection and per-device execution counts.
Counts describe test files and configured functional cases, not pytest test counts.
Selection reasons, validation gaps, full identity and JSON are expandable below;
the unchanged `ci-plan` artifact remains the machine-readable source. Functional
job display names omit execution specs; those specs still bind evidence to the plan.

`commands.py` handles `/ci run <platform>`, `/ci run all` and
`/ci rerun-job <job-id>`. Repository writers and human PR authors are authorized.
Authors must have an approved, completed current-head platform CI before dispatch;
a selected rerun must identify a latest-attempt failed job from that exact PR source.
PR head and run attempt are rechecked before mutation. Requested CI retains its
own identity and does not replace the ordinary required CI gate.

`github_api.py` uses only Python's standard library. It reads `GH_TOKEN` and
`GITHUB_API_URL`, paginates REST lists, strips authentication on cross-origin log
redirects, limits response size/time, and never automatically retries a POST.
Controllers are checked out from the default branch; candidate code is data only.
Planner, runner, comparator and evidence contracts are unchanged by this migration.

`infra_retry.py` applies a code-reviewed recovery policy without a repository
variable. Once this workflow and controller are merged into the default branch,
only first-attempt default-branch push/scheduled CI with proven exhausted runtime
artifact transfer failures can request one failed-jobs retry. The workflow filters
out ineligible events before starting a runner; the controller independently
revalidates fresh API state before any mutation. Timestamp scoping,
corruption/authorization rejection, mixed-failure rejection and final
attempt/default-branch SHA checks are retained. PRs, dispatches and test/performance
failures are ineligible. The existing prepare, test and evidence gates still run
on the retried attempt; failure is never converted to success by the controller.

No `CI_INFRA_RETRY_ENABLED` setting, custom token or extra dependency installation
is required. Any previously configured value of that variable is ignored. GitHub
provides the workflow token under the declared `actions: write` permission;
repository/organization policy must still permit that permission. This policy
becomes active through normal default-branch code review and merge, not merely by
updating a fork PR. To suspend it, disable the retry workflow or review a code
change to its eligibility conditions. The bounded retries already inside runtime
recovery remain unchanged and run before this final, at-most-once workflow retry.
