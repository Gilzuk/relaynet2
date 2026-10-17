"""Seed and record a fresh child process before importing an experiment driver."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    evidence = Path(args.evidence)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np

    random.seed(job['seed'])
    np.random.seed(job['seed'])
    torch = None
    if job.get('needs_torch'):
        import torch
        torch.manual_seed(job['seed'])
        torch.set_num_threads(job['threads'])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def snapshot(label):
        state = np.random.get_state()
        record = {
            'captured_utc': datetime.now(timezone.utc).isoformat(),
            'source_commit': job['source_commit'], 'bootstrap_seed': job['seed'],
            'python': sys.version, 'platform': platform.platform(),
            'packages': dict(sorted((d.metadata['Name'], d.version)
                                    for d in importlib.metadata.distributions() if d.metadata['Name'])),
            'environment': {k: os.environ.get(k) for k in (
                'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'CUBLAS_WORKSPACE_CONFIG', 'CODED_METRIC', 'CODED_TRIALS', 'CMM_TRIALS', 'CMM_SNRS')},
            'python_state': random.getstate(),
            'numpy_legacy_state': [state[0], state[1].tolist(), *state[2:]],
            'torch': None,
            'scope': 'global bootstrap; driver-local generators and seed overrides remain part of its protocol',
        }
        if torch is not None:
            record['torch'] = {
                'version': torch.__version__, 'cpu_state_hex': bytes(torch.get_rng_state().tolist()).hex(),
                'cuda_state_hex': [bytes(s.tolist()).hex() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                'initial_seed': torch.initial_seed(), 'threads': torch.get_num_threads(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'build_config': torch.__config__.show(), 'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
            }
        (evidence / f'rng_{label}.json').write_text(json.dumps(record, indent=2) + '\n')

    snapshot('before')
    sys.argv = [job['script'], *job['args']]
    try:
        runpy.run_path(str(root / job['script']), run_name='__main__')
    finally:
        snapshot('after')


if __name__ == '__main__':
    main()
