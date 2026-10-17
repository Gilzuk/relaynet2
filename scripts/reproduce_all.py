"""Run the registered experiment producers in an isolated worktree and retain evidence.

Default: print the plan only. Historical suite records are archive references,
not rerunnable commands. Execution uses committed HEAD, a recorded RNG bootstrap
(default 42), declared full budgets, and per-step immutable output snapshots.
Missing/stale output or failed verification returns 1; changed output bytes
return 2 for scientific review. A byte match is not statistical equivalence.
Historical learned runs with missing RNG/checkpoints are not exact-replay claims.

Dependency inference covers literal paths and shared writers only. Partial runs
may read archived upstream data; this is recorded in each step. Some original
drivers do not retain weights. Use reproduce_round2.py for the focused suite's
richer per-field comparisons and checkpoint evidence. Full execution is costly.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CANON_BUDGET = ["--num-trials", "10", "--bits-per-trial", "10000", "--snr-min", "0",
                "--snr-max", "20", "--snr-step", "2", "--skip-relays", "cgan", "--retrain"]
MATCHED_ENV = {"CODED_METRIC": "matched", "CODED_TRIALS": "30"}

# name -> (script, args, env, outputs); declared order is the fallback order.
EXTRA = [
    ("Canonical comparison (Table 5.2)", "run_experiments.py", ["--exp", "7.10"] + CANON_BUDGET, {},
     ["results/modulation/qpsk_rayleigh.json"]),
    ("Equal-budget comparison (Table 5.3)", "run_experiments.py", ["--exp", "7.8"] + CANON_BUDGET, {},
     ["results/normalized_3k/3k_rayleigh.json"]),
    ("Coded block DF, K sweep (unweighted)", "coded_df_experiment.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded K sweep, QPSK (unweighted)", "coded_k_sweep_qpsk.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded K sweep, 16-QAM (unweighted)", "coded_k_sweep_qam16.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded learned MLP relay (unweighted)", "coded_learned_relay.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded Mamba relay (unweighted)", "coded_mamba_relay.py", [], {}, ["results/coded_df_experiment.json"]),
    ("Coded high-SNR paired re-test", "coded_high_budget_test.py", [], {}, ["results/coded_high_budget_test.json"]),
    ("Coded error location (unweighted)", "coded_error_mechanism.py", [], {}, ["results/coded_error_mechanism.json"]),
    ("Coded reliable regime (unweighted)", "coded_reliable_regime.py", [], {}, ["results/coded_reliable_regime.json"]),
    ("Coded soft-decision relaying", "coded_soft_decision.py", [], {}, ["results/coded_soft_decision.json"]),
    ("Coded equal throughput (unweighted)", "coded_latency_throughput.py", [], {}, ["results/coded_latency_throughput.json"]),
    ("Coded rate adaptation (unweighted)", "coded_rate_adaptation.py", [], {}, ["results/coded_rate_adaptation.json"]),
    ("Coded readiness budget (unweighted)", "coded_latency_capacity.py", [], {}, ["results/coded_latency_capacity.json"]),
    ("Coded K sweep, QPSK, matched metrics", "coded_k_sweep_qpsk.py", [], MATCHED_ENV,
     ["results/coded_df_experiment_matched.json"]),
    ("Coded equal throughput, matched metrics", "coded_latency_throughput.py", [], MATCHED_ENV,
     ["results/coded_latency_throughput_matched.json"]),
    ("Coded readiness budget, matched metrics", "coded_latency_capacity.py", [], MATCHED_ENV,
     ["results/coded_latency_capacity_matched.json"]),
]
ARCHIVE_SCRIPTS = {
    'scripts/audit_rng_reproducibility.py': 'Historical frozen-source RNG audit; requires an explicit source and output directory.',
    'scripts/reproduce_round2.py': 'Historical suite manifest, not a standalone producer of that dated archive.',
    'scripts/reproduction_original_control.py': 'Historical retained-checkpoint control; requires its original evidence and replay snapshot.',
}
EVIDENCE_ROOT = os.path.join(ROOT, 'results', 'reproduce_all')
STUDY_ENV = ('CODED_METRIC', 'CODED_TRIALS', 'CMM_TRIALS', 'CMM_SNRS')


def slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def source(script):
    return (Path(ROOT) / script).read_text(encoding='utf-8')


def code_strings(script):
    """Literal strings, excluding comments and docstrings, used only as input hints."""
    tree = ast.parse(source(script))
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def needs_torch(script, seen=None):
    """Conservative local-import closure; optional Torch imports also count."""
    seen = set() if seen is None else seen
    if script in seen:
        return False
    seen.add(script)
    tree = ast.parse(source(script))
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                parents = Path(script).parent.parts
                prefix = '.'.join((*parents[:len(parents) - node.level + 1], prefix)).strip('.')
            modules = [prefix, *(f'{prefix}.{a.name}' for a in node.names)]
        for module in modules:
            if module == 'torch' or module.startswith('torch.'):
                return True
            for suffix in ('.py', '/__init__.py'):
                local = module.replace('.', '/') + suffix
                if (Path(ROOT) / local).is_file() and needs_torch(local, seen):
                    return True
    return False


def build_plan():
    """Account for all registry outputs; historical archives are reference-only."""
    from provenance_audit import REGISTRY
    steps = [dict(name=n, script=s, args=list(a), env=dict(e), outputs=list(o), kind='experiment')
             for n, s, a, e, o in EXTRA]
    for name, (script, outputs, _art) in REGISTRY.items():
        env = {}
        if any(o.endswith('_matched.json') for o in outputs):
            if 'matched' not in script:
                env = dict(MATCHED_ENV)
            elif 'CODED_METRIC' in source(script):
                env = {'CODED_METRIC': 'matched'}
        if script in ARCHIVE_SCRIPTS:
            steps.append(dict(name=name, script=script, args=[], env={}, outputs=list(outputs),
                              kind='archive_reference', reason=ARCHIVE_SCRIPTS[script]))
            continue
        if script == 'rare_event_fixed_budget.py':
            for output in outputs:
                replicate = 'replicate' in output
                steps.append(dict(name=name + (' (replicate)' if replicate else ' (original)'),
                                  script=script, args=['--src-base', '7000' if replicate else '1000',
                                                       '--out', output], env={}, outputs=[output], kind='experiment'))
            continue
        dup = next((s for s in steps if s['script'] == script and not s['args'] and s['env'] == env), None)
        if dup:
            dup['outputs'] = sorted(set(dup['outputs']) | set(outputs))
            dup['name'] += f'; {name}'
        else:
            kind = 'archive_analysis' if script in ('scripts/audit_round2_models.py',
                                                   'scripts/assess_round2_reruns.py') else 'experiment'
            steps.append(dict(name=name, script=script, args=[], env=env, outputs=list(outputs), kind=kind))
    writers = {}
    for i, step in enumerate(steps):
        step['needs_torch'] = step['kind'] != 'archive_reference' and needs_torch(step['script'])
        step['after'] = set()
        if step['kind'] != 'archive_reference':
            for output in step['outputs']:
                writers.setdefault(output, []).append(i)
    # Shared JSON writers must run in declaration order. A reader runs after the
    # last producer. Archive records never become executable dependencies.
    for indices in writers.values():
        for previous, current in zip(indices, indices[1:]):
            steps[current]['after'].add(previous)
    basenames = {}
    for output in writers:
        basenames.setdefault(Path(output).name, []).append(output)
    for i, step in enumerate(steps):
        if step['kind'] in ('archive_reference', 'archive_analysis'):
            continue
        strings = code_strings(step['script'])
        for output, indices in writers.items():
            if output in step['outputs']:
                continue
            if output in strings or (len(basenames[Path(output).name]) == 1 and Path(output).name in strings):
                step['after'].add(indices[-1])
    order, done, pending = [], set(), list(range(len(steps)))
    while pending:
        ready = [i for i in pending if steps[i]['after'] <= done]
        if not ready:
            raise RuntimeError('dependency cycle among: ' + ', '.join(steps[i]['name'] for i in pending))
        i = ready[0]
        order.append(i)
        done.add(i)
        pending.remove(i)
    plan = []
    for i in order:
        step = dict(steps[i])
        step['after'] = [steps[j]['name'] for j in sorted(step['after'])]
        step['dependency_method'] = 'literal-path hints plus ordered shared-output writers; dynamic inputs not inferred'
        plan.append(step)
    return plan


def select(plan, only=None, skip=None, no_torch=False):
    keep = [dict(s) for s in plan if (not only or re.search(only, f"{s['name']} {s['script']}", re.I))
            and (not skip or not re.search(skip, f"{s['name']} {s['script']}", re.I))
            and (not no_torch or not s['needs_torch'])]
    names = {s['name'] for s in keep}
    for step in keep:
        step['archived_dependencies'] = [dep for dep in step['after'] if dep not in names]
    return keep


def command(step):
    env = ' '.join(f'{k}={v}' for k, v in step['env'].items())
    return (env + ' ' if env else '') + shlex.join(['python3', step['script'], *step['args']])


def print_plan(plan):
    for k, step in enumerate(plan, 1):
        print(f"[{k:2d}] {step['name']} [{step['kind']}]" + (' (needs PyTorch)' if step['needs_torch'] else ''))
        print('     ' + (step['reason'] if step['kind'] == 'archive_reference' else '$ ' + command(step)))
        for output in step['outputs']:
            print(f'       -> {output}')
        if step.get('archived_dependencies'):
            print('       uses committed inputs from omitted producers: ' + '; '.join(step['archived_dependencies']))
    print(f'\n{len(plan)} entries. Nothing was run; add --run to execute. Default bootstrap seed: 42.')


def git(*args, cwd=None):
    return subprocess.run(['git', *args], cwd=cwd or ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def committed_sha256(commit, path):
    result = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def safe_path(root, relative):
    rel = Path(relative)
    root = Path(root).resolve()
    path = root / rel
    if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root):
        raise ValueError(f'unsafe artifact path: {relative}')
    if path.is_symlink():
        raise ValueError(f'symlink artifact is not supported: {relative}')
    return path


def stamp(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def artifact_stamps(worktree):
    """Track driver-written results and saved weights, including ignored files."""
    root = Path(worktree)
    return {str(path.relative_to(root)): stamp(path)
            for folder in ('results', 'weights', 'reproduction_checkpoints')
            for path in (root / folder).rglob('*') if path.is_file() and not path.is_symlink()}


def compare(commit, worktree, path, outdir, *, fresh=True, executed=True):
    regen = safe_path(worktree, path)
    ref = committed_sha256(commit, path)
    record = {'path': path, 'committed_sha256': ref}
    if not regen.is_file():
        return dict(record, status='MISSING')
    dst = safe_path(outdir, path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(regen, dst)
    new = sha256_file(dst)
    if not executed:
        status = 'ARCHIVE_REFERENCE'
    elif not fresh:
        status = 'NOT_REGENERATED'
    else:
        status = 'NEW' if ref is None else ('BYTE_MATCH' if new == ref else 'BYTES_DIFFER_REVIEW')
    return dict(record, status=status, observed_sha256=new)


def run(plan, evidence_dir, worktree_dir=None, keep_worktree=False, verify=True, *, seed=42, threads=2):
    """Run committed code only; fail closed on stale/missing outputs and verifier errors."""
    if not 0 <= seed < 2**32 or threads < 1:
        raise ValueError('seed must be in [0, 2**32); threads must be positive')
    # A dirty script/registry cannot silently plan one version and run another.
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Commit tracked changes before --run; the plan and execution must use the same HEAD.')
    commit = git('rev-parse', 'HEAD')
    evidence = Path(evidence_dir).resolve()
    wt = Path(worktree_dir or tempfile.mkdtemp(prefix='reproduce_all_')).resolve()
    if wt == Path(ROOT).resolve() or evidence.is_relative_to(wt) or wt.is_relative_to(evidence):
        raise ValueError('evidence and disposable worktree must be separate directories')
    if wt.exists() and any(wt.iterdir()):
        raise ValueError(f'worktree directory {wt} is not empty')
    for step in plan:
        safe_path(wt, step['script'])
        for output in step['outputs']:
            safe_path(wt, output)
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / 'steps').mkdir()
    meta = {'source_commit': commit, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'worktree': str(wt), 'steps': plan,
            'rng_policy': {'bootstrap_python_numpy_torch_seed': seed, 'threads': threads,
                           'experiment_local_seeds': 'preserved; may override the bootstrap'},
            'comparison_scope': 'file bytes, not numerical or statistical equivalence; archive entries are not reruns',
            'dependency_scope': 'literal-path hints; omitted/dynamic inputs may remain committed archival inputs',
            'retention_scope': 'declared outputs, changed files in results/weights/reproduction_checkpoints, logs and RNG states; unsaved in-memory weights cannot be recovered',
            'verify_status': 'PENDING' if verify else 'NOT_REQUESTED'}
    summary, comparisons, outcomes, produced = [], [], {}, set()
    added = False
    preserve = keep_worktree

    def save():
        write_json(evidence / 'plan.json', meta)
        write_json(evidence / 'summary.json', summary)
        write_json(evidence / 'comparison.json', comparisons)

    save()
    try:
        git('worktree', 'add', '--detach', str(wt), commit)
        added = True
        for k, step in enumerate(plan, 1):
            dirname = f"{k:03d}-{slug(step['name'])[:70]}"
            stepdir = evidence / 'steps' / dirname
            stepdir.mkdir()
            log = stepdir / 'run.log'
            # Literal-path inference cannot prove independence after a failure.
            # Stop executing later producers rather than risk using stale inputs.
            blocked = [name for name, status in outcomes.items() if status not in ('COMPLETED', 'ARCHIVE_REFERENCE')]
            rec = {'step': step['name'], 'command': command(step), 'kind': step.get('kind', 'experiment'),
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'returncode': None,
                   'evidence': str(stepdir.relative_to(evidence)), 'archived_dependencies': step.get('archived_dependencies', [])}
            print(f"[{k}/{len(plan)}] {step['name']} ... ", end='', flush=True)
            t0 = time.monotonic()
            cmp = []
            if step.get('kind') == 'archive_reference':
                cmp = [compare(commit, wt, output, stepdir / 'outputs', executed=False) for output in step['outputs']]
                rec['status'] = 'ARCHIVE_REFERENCE' if all(c['status'] == 'ARCHIVE_REFERENCE' for c in cmp) else 'MISSING'
                rec['reason'] = step['reason']
            elif blocked:
                rec.update(status='BLOCKED', blocked_by=blocked,
                           reason='Earlier step failed; dynamic dependencies cannot be ruled out.')
            else:
                artifacts_before = artifact_stamps(wt)
                before = {}
                for output in step['outputs']:
                    path = safe_path(wt, output)
                    # Remove committed outputs before their first producer. Shared
                    # writers keep only the preceding producer's freshly written file.
                    if output not in produced:
                        path.unlink(missing_ok=True)
                    before[output] = stamp(path)
                env = dict(os.environ)
                for key in STUDY_ENV:
                    env.pop(key, None)
                env.update({'PYTHONHASHSEED': str(seed), 'OMP_NUM_THREADS': str(threads),
                            'OPENBLAS_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads),
                            'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONPATH': str(wt)})
                env.update(step['env'])
                spec = dict(step, seed=seed, threads=threads, source_commit=commit)
                write_json(stepdir / 'job.json', spec)
                argv = [sys.executable, str(wt / 'scripts/reproduce_all_worker.py'),
                        '--job', str(stepdir / 'job.json'), '--evidence', str(stepdir)]
                with log.open('w', encoding='utf-8') as fh:
                    fh.write(f'$ {command(step)}\n')
                    fh.flush()
                    result = subprocess.run(argv, cwd=wt, env=env, stdout=fh, stderr=subprocess.STDOUT)
                rec['returncode'] = result.returncode
                cmp = [compare(commit, wt, output, stepdir / 'outputs',
                               fresh=stamp(safe_path(wt, output)) != before[output]) for output in step['outputs']]
                bad = any(c['status'] in ('MISSING', 'NOT_REGENERATED') for c in cmp)
                rec['status'] = 'FAILED' if result.returncode != 0 or bad else 'COMPLETED'
                if result.returncode != 0:
                    for item in cmp:
                        if item['status'] not in ('MISSING', 'NOT_REGENERATED'):
                            item['status'] = 'PARTIAL_OUTPUT'
                if rec['status'] == 'COMPLETED':
                    produced.update(step['outputs'])
                auxiliary = []
                for path, state in artifact_stamps(wt).items():
                    if path in step['outputs'] or state == artifacts_before.get(path):
                        continue
                    original = safe_path(wt, path)
                    dest = safe_path(stepdir / 'extra_outputs', path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, dest)
                    auxiliary.append({'path': path, 'sha256': sha256_file(dest),
                                      'evidence': str(dest.relative_to(evidence)),
                                      'step_status': rec['status']})
                rec['extra_outputs'] = auxiliary
            rec.update(seconds=round(time.monotonic() - t0, 3), finished_utc=datetime.now(timezone.utc).isoformat())
            outcomes[step['name']] = rec['status']
            comparisons.extend(dict(c, step=step['name'], step_status=rec['status'],
                                    evidence=str((stepdir / 'outputs' / c['path']).relative_to(evidence))) for c in cmp)
            summary.append(rec)
            print(f"{rec['status']} ({rec['seconds']:.1f}s); " + ', '.join(c['status'] for c in cmp))
            save()
        failures = any(r['status'] not in ('COMPLETED', 'ARCHIVE_REFERENCE') for r in summary)
        if verify and failures:
            meta['verify_status'] = 'BLOCKED_BY_FAILED_STEPS'
        elif verify:
            with (evidence / 'verify.log').open('w') as fh:
                result = subprocess.run([sys.executable, 'verify_thesis_tables.py'], cwd=wt,
                                        stdout=fh, stderr=subprocess.STDOUT)
            meta.update(verify_returncode=result.returncode, verify_status='PASSED' if result.returncode == 0 else 'FAILED')
        # A file difference is a review request, not a crashed experiment or an
        # inference that the historical result was incorrect.
        differing = any(c['status'] in ('BYTES_DIFFER_REVIEW', 'NEW') for c in comparisons)
        meta['exit_code'] = 1 if failures or meta['verify_status'] == 'FAILED' else (2 if differing else 0)
    except BaseException as exc:
        meta.update(error=f'{type(exc).__name__}: {exc}', exit_code=1)
        preserve = True
        raise
    finally:
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if added and not preserve:
            result = subprocess.run(['git', 'worktree', 'remove', '--force', str(wt)], cwd=ROOT, capture_output=True, text=True)
            meta['worktree_cleanup'] = 'REMOVED' if result.returncode == 0 else result.stderr.strip()
        else:
            meta['worktree_cleanup'] = 'RETAINED' if added else 'NOT_CREATED'
        save()
    return summary, comparisons, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='store_true', help='execute in a detached worktree; default is a dry run')
    parser.add_argument('--only', help='regex selecting entries; omitted dependencies use committed inputs')
    parser.add_argument('--skip', help='regex excluding entries')
    parser.add_argument('--no-torch', action='store_true', help='exclude scripts with direct or local transitive Torch imports')
    parser.add_argument('--json', action='store_true', help='print the plan as JSON, without running')
    parser.add_argument('--run-id', help='new evidence folder name')
    parser.add_argument('--evidence-dir')
    parser.add_argument('--worktree-dir')
    parser.add_argument('--keep-worktree', action='store_true')
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Python/NumPy/Torch bootstrap seed (default: 42)')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    plan = select(build_plan(), args.only, args.skip, args.no_torch)
    if not plan:
        parser.error('no entries selected')
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.run:
        print_plan(plan)
        return 0
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + git('rev-parse', '--short', 'HEAD')
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        parser.error('--run-id must be one folder name')
    evidence = args.evidence_dir or os.path.join(EVIDENCE_ROOT, run_id)
    try:
        summary, comparison, meta = run(plan, evidence, args.worktree_dir, args.keep_worktree,
                                         verify=not args.no_verify, seed=args.seed, threads=args.threads)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    counts = {}
    for item in comparison:
        counts[item['status']] = counts.get(item['status'], 0) + 1
    print(f"\nEvidence: {evidence}\nOutput comparisons: {counts}\nVerifier: {meta['verify_status']}")
    return meta['exit_code']


if __name__ == '__main__':
    sys.exit(main())
