# Round-2 reproduction runs

PR [183](https://github.com/Gilzuk/relaynet2/pull/183) was merged at `42e3e386e0b6d1c137a84cf609b51e9581e7e69a` after CI passed. This separate branch adds fresh evidence. It does not silently replace the published thesis or its inputs.

Current evidence and investigation: [results/reproduction/README.md](../results/reproduction/README.md). The initial setup attempt failed before simulation; the corrected local run `20261017-round2-v2` then lost its session after 09:07:25 UTC. Four completed checks matched and 16 training seeds/checkpoints were recovered. Both attempts remain preserved. Do not interpret their old heartbeat or PID as proof of continued execution.

All four hosted jobs in run 39024026594 have now completed. Their complete evidence is committed at `dc7c02c00cbebdfb566362a810ab4b45cb6abbe0`; the numerical comparator flags learned differences for review, with exact classical controls. These runs supplement the training-realization and multi-seed evidence; they do not, from numerical disagreement alone, invalidate original measurements. The sequence study repeats the same three seed IDs, and the coded study retains its initialization seed; not all differences can be attributed to changed seeds. The [post-run assessment](round2_rerun_manuscript_update_2026-10-17.md) explains the manuscript updates and fresh paired tests. No new run was launched for this assessment.

## Persistent hosted execution

[`.github/workflows/round2-reproduction.yml`](../.github/workflows/round2-reproduction.yml) runs the four unfinished jobs independently on GitHub Actions when the declared workflow/runner/dependency files change on PR 184's branch. It is restricted to this repository and `experiments/round2-reproduction-20261017`; it never cancels an existing run on a new push. Every job checks out the exact PR head SHA and records its actual environment. Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and CPU Torch 2.8.0 are declared; CPU/hardware differences remain visible. Training, evaluation budgets and comparison tolerances are unchanged.

Each experiment has a 5.5-hour timeout inside a 350-minute workflow job, leaving time to harvest failures. Fresh IDs include workflow run, attempt and experiment. The complete sequence study is repeated on the new host; seeds are not spliced across environments. Upload runs even when the comparator requires review. An independent final job copies the artifacts to versioned `results/reproduction/gha-*` directories, verifies their source revisions, refuses to overwrite previous evidence, records missing jobs explicitly, and commits only evidence to the reproduction branch. Artifacts are also retained for 90 days. There is no automatic merge or replacement of published thesis inputs. A nonzero comparison exit can mean a learned result needs review; inspect the JSON status rather than treating every red check as a setup failure.

The local launcher below remains useful on a machine whose execution lifetime the author controls. The hosted chat execution session did not provide that persistence.
## Launch and monitor

Use Python with NumPy, SciPy, Matplotlib and CPU PyTorch installed. The first execution environment uses Python 3.12.14 and PyTorch 2.8.0+cpu; the run's `environment.json` records the actual complete package set. These are new validation versions, not recovered historical versions.

```bash
python scripts/reproduce_round2.py --list
python scripts/reproduce_round2.py --run-id 20261017-round2 --background --threads 2
cat results/reproduction/20261017-round2/status.json
tail -f results/reproduction/20261017-round2/sequence_memory/run.log
```

The launcher returns the coordinator PID and a status path. It detaches from the launching shell. This does not guarantee survival of host shutdown; inspect the recorded PID and heartbeat before assuming an old `RUNNING` entry is live. Runs are sequential, with two numerical-library threads per worker; the existing coded driver also forks six SNR workers. Each job has an eight-hour timeout, recorded explicitly. Training jobs can take hours. No GPU is requested, no reduced budget is substituted, and no failed seed is dropped. A run ID cannot be reused or overwrite existing evidence. To repeat a failed job, use a new ID and `--jobs JOB`; retain the earlier failure.

The source commit is frozen before execution. Each job has a disposable detached worktree; the target output is removed first to prevent accidentally accepting a cached result. Worktrees are removed only after outputs are harvested. Published reference files in the main checkout are never written by the workers and their hashes are checked afterward.

## Scope

| Job | What is regenerated | Comparison |
|---|---|---|
| `common_target_analysis` | All common-target penalty summaries, including minimum-size/seed-spread conclusions | Full derived JSON, ignoring only the top-level source revision |
| `canonical_paired_analysis` | Paired tests from stored canonical trials | Full derived JSON |
| `canonical_af_df` | Fresh canonical AF/DF trials at the original grid and budget | Both original canonical files' mean and per-trial BER arrays |
| `factorized_mlse` | Original full joint/factorized decision-equivalence study | Full scientific JSON, ignoring only source revision |
| `sequence_memory` | All three channels, four architectures and three initialization seeds of Table 6.7 | Original per-seed BER curves and baseline; fresh common-target penalties, bounds, rankings and checkpoints |
| `coded_matched` | Original matched-coded experiment: 30 trials × 500 frames at six SNRs | Full JSON except elapsed time; classical controls checked separately |
| `canonical_native`, `canonical_3k` | Original 100-epoch protocols with a new declared PyTorch seed | Scientific BER arrays; AF/DF controls checked separately; all learned checkpoints retained |

The first two jobs are saved-data reanalysis, not new Monte Carlo evidence. The suite does not retrain the entire minimum-size grid, rerun adaptive rare-event tails, the pilot sweep, full-window/bidirectional Mamba controls or every secondary study. It does not add matched Mamba. Those limits are repeated in the generated report rather than concealed by an “all reproduced” label.

## Evidence and decisions

Future runs declare and enforce an RNG policy for every job: Python, global
NumPy and Torch bootstrap seed 20261017, `PYTHONHASHSEED=0`, CPU execution,
fixed thread counts and deterministic Torch algorithms. Each study's original
local seeds remain explicit in the policy. Global states at job entry/exit
are archived under `outputs/reproduction_checkpoints/rng/`; local-generator
seeds and the original call order support replay from the beginning. These
snapshots do not implement mid-job resume. See the
[historical RNG audit and full-budget seed-42 control](rng_reproducibility_audit_2026-10-17.md).
The already launched hosted run 39024026594 uses frozen source `c0394f0` and
continues with its original settings; the new flags/snapshots apply only to
later runs. Same-host seed-42 MLP repeats matched exactly, but cross-environment
identity and historical learned-result recovery remain unestablished.

Every run retains `manifest.json`, `environment.json`, `status.json` and `REPORT.md`. Each job retains its exact command and environment overrides, source revision, start/end times, exit status, log, reference JSON snapshots with SHA-256 hashes and immutable GitHub links, raw new outputs, and `comparison.json`. Checkpoints from new canonical and Table 6.7 training are included under the job's output folder. Table 6.7 outputs are updated after every completed seed in the worktree; final/failed-job harvesting preserves the available outputs.

`MATCH` means the declared scientific fields agree (roundoff tolerance only: relative 1e-12, absolute 1e-14). It supplies `matching_reference` paths back to the relevant original JSON. It does not imply byte identity, statistical equivalence, or verification of untested claims. Deterministic/control changes are `MISMATCH`. Changed learned results are `RETRAINING_DIFFERENCE_REQUIRES_REVIEW`, with every differing field retained. Missing output, exceptions and timeouts are failures, never matches. A changed ranking or failed target is explicit in the new per-target summaries.

The original canonical PyTorch RNG state and learned checkpoints are missing. The two canonical training jobs therefore use the declared new seed 20261017, while retaining the original driver/data-seed settings; they replicate the protocol, not the lost training realization. Their checkpoints and pre-training Torch RNG state are now saved. A single new training realization cannot quantify historical training variance.

## Inconsistencies found while preparing the runner

1. `requirements.txt` said only cGAN required Torch; canonical learned replication also needs Torch for the MLP/Hybrid path, VAE and sequence models. Corrected the dependency guidance.
2. The environment doctor still claimed it verified every number and that the unknown-channel run took about 40 minutes. Corrected to the verifier's declared subset and driver-dependent, potentially hours-long adaptive budgets.
3. Historical sequence-driver `db`, `best_db` and `spread` fields use the superseded rule that drops missed targets. They remain historical provenance, not authorities for corrected Table 6.7. The new worker preserves the original training/evaluation call order but computes and retains `penalty_common` per target and the corrected seed summaries. The authoritative published comparison is `results/penalty_common_targets.json`.
4. The native canonical driver does not save all learned weights after fresh training. The new worker captures each network before evaluation without consuming RNG draws or changing its selected parameters. Both native and 3K runs fail if any expected learned model/checkpoint is missing.

Discrepancies are investigated against configuration, code and source data. No threshold is tuned to obtain a match, no original result is replaced automatically, and a change to a thesis claim requires new supporting evidence and a separate reviewed correction.
