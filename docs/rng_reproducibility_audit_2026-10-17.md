# RNG audit and seed-42 control — 17 October 2026

The historical canonical `--seed 42` option did not seed PyTorch model
initialization or minibatch permutations. A full-budget native MLP control
confirms that explicitly seeding all RNGs before construction makes this
example repeat exactly on the current host. This does not recover the
historical learned weights or establish reproducibility of every architecture.

## What the historical code actually seeded

Native canonical source:
[`b4059e615ef8e1f9985bb3b61fed987eefa62945`](https://github.com/Gilzuk/relaynet2/tree/b4059e615ef8e1f9985bb3b61fed987eefa62945).
Published native data:
[`69a470cf5931453bc11d67a99943aa56ae565757`](https://github.com/Gilzuk/relaynet2/blob/69a470cf5931453bc11d67a99943aa56ae565757/results/modulation/qpsk_rayleigh.json).
The native and 3K paths were inspected at their recorded source revisions
(`b4059e6` and `4cf9753`), not inferred from a current CLI default.

| Path | Established RNG behavior | Consequence |
|---|---|---|
| Canonical CLI / `WeightManager` | `--seed` defaults to 42; used for training arguments and checkpoint directory naming | A directory named `seed42` does not prove that initialization used 42 |
| Canonical `MinimalGenAIRelay` | Models are constructed before `train(seed=42)`; that call seeds global NumPy only | Torch initialization and `torch.randperm` have no explicit seed on this path; fallback NumPy initialization also occurs before the training seed |
| Canonical training data | Generator seed is `42 + int(SNR)`; MLP uses 47, 52 and 57 at 5, 10 and 15 dB | Data repeatability is separate from weight/optimizer repeatability |
| Canonical evaluation | Trial/source seeds 0–9 at every SNR | The classical controls can reproduce even when learned weights differ |
| Sequence-memory study | Torch seeded with 0, 1, 2 before construction and again before training; training bits use local NumPy generators 1000, 1001, 1002 | Preserve the original baseline/training/evaluation order and channel state; these historical runs already contain explicit training seeds |
| Matched-coded study | Pure NumPy classifier: initialization 0, training-data bits/fading/noise 0, training shuffle 42; paired evaluation `800000 + 1000*SNR + trial` | This path does not depend on Torch model initialization |

Other checkpoint scripts contain `torch.manual_seed`, but their presence does
not establish a seed for the canonical 7.10/7.8 entry points. An external
historical seeding step cannot be ruled out; no such step, original Torch state
or matching canonical checkpoint has been recovered.

## Full-budget example actually executed

[`scripts/audit_rng_reproducibility.py`](../scripts/audit_rng_reproducibility.py)
runs four fresh processes against clean historical source. It preserves the
native model construction order, including the subsequently excluded cGAN,
then trains only the first 169-parameter MLP (5→24→1): 25,000 requested
samples, 24,987 training windows, 100 epochs, batch 32, Adam with learning rate
0.01, final-epoch weights. Evaluation uses QPSK/Rayleigh, 0–20 dB in 2 dB
steps, ten trials and 10,000 bits per trial: 110 per-trial BER values.

Both conditions use the same source, host, CPU Torch 2.8.0, NumPy 2.3.5,
Python 3.12.14, two numerical-library threads and fixed Python hash seed 0.
The explicit condition sets Python, NumPy and Torch to 42 before model
construction and enables deterministic Torch algorithms. The legacy condition
retains the historical training seed argument without adding those settings.
This is a combined reproducibility control, not an experiment isolating the
effect of each flag individually.

| Comparison between two fresh runs | Legacy training seed only | All RNGs seeded before construction |
|---|---:|---:|
| Training-data hashes identical | Yes | Yes |
| Initial weight tensors identical | No | Yes |
| Final weight tensors identical | No | Yes |
| Differing trial-BER values out of 110 | 102 | 0 |
| Maximum absolute mean-BER difference | 0.00070 | 0 |

The fully seeded run still differs from the published native MLP result:
maximum absolute mean-BER difference **0.00150**. Seed 42 is a declared new
realization, not a recovered historical Torch seed. Exact repetition of this
one model on this host does not prove cross-platform identity or statistical
equivalence to the historical results.

Evidence: [comparison.json](../results/reproduction/20261017-rng-seed42/comparison.json),
[manifest.json](../results/reproduction/20261017-rng-seed42/manifest.json).
The directory retains all four results/logs, initial and final weights,
training-data hashes, full Python/NumPy/Torch states at three checkpoints,
the published reference snapshot, exact commands and the executed driver.
All 47 artifact hashes and the executed-driver hash were verified.

To repeat on a fresh clone with the declared dependencies installed:

```bash
git worktree add --detach ../relaynet2-native-rng b4059e615ef8e1f9985bb3b61fed987eefa62945
python scripts/audit_rng_reproducibility.py --source-dir ../relaynet2-native-rng --output-dir results/reproduction/rng-seed42-repeat
```

Use a new output directory; the script refuses to overwrite evidence.

## Future suite runs

Every job now declares an RNG policy in its manifest. The worker seeds Python,
global NumPy and Torch with **20261017** before experiment imports/construction,
then preserves each study's explicit local seed schedule above. Torch
deterministic algorithms are required; an unsupported nondeterministic
operation raises an error. The launcher already fixes `PYTHONHASHSEED=0`,
CPU execution and numerical-library thread counts.

The worker saves full global Python/NumPy/Torch states at job entry and exit,
plus policy, source revision, Torch version and thread count. Local generator
seeds are recorded in the policy; their intermediate states are not captured.
This supports replay from the beginning with the same execution order, not
arbitrary mid-job resumption. Canonical checkpoints and pre-construction Torch
state remain retained separately.

These additions apply to future source revisions. Hosted run
[`39024026594`](https://github.com/Gilzuk/relaynet2/actions/runs/39024026594)
continues at frozen source `c0394f0`; its existing canonical seed is already
20261017. No active job was cancelled, changed or restarted. Strict deterministic
flags and the new all-job state snapshots cannot be retroactively claimed for
that run.

Ten focused runner/RNG/provenance tests passed. The archived interrupted run's
provenance is checked against its independently pinned producer `43ce4a6`,
rather than treating a later runner revision as its producer. A mismatched
source declaration fails the audit; current thesis inputs retain their normal
staleness checks. Published scientific data and the thesis are unchanged.

PyTorch's [version-matched reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
explains why identical seeds do not guarantee identical outputs across library
versions, platforms or CPU/GPU execution.
