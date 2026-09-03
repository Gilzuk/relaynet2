"""Mutation tests for verify_thesis_tables.py.

A verifier that reports "OK" because it examined nothing is indistinguishable,
from the outside, from one that examined everything and found no problem. That
gap is not hypothetical: it hid a parameter-count check that validated hardcoded
constants against their own arithmetic and so could never fail, two unguarded
`find()` calls that would have sliced the wrong region of the document, and an
earlier joint-latency check that read three of ten rows while printing OK.

The only way to know a check can fail is to make it fail. Each case below
perturbs one published number in a scratch copy of the thesis and asserts that
the named check flags it. If a check ever stops catching its own mutation, the
check is broken even when the thesis is fine.

`verify_thesis_tables.py` takes `--tex` to point at a different chapters/
directory, so nothing here touches the real thesis.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "thesis" / "chapters"

# (check name, file, exact published text to perturb, replacement)
# Each replacement must be a value the source data does NOT support.
MUTATIONS = [
    ("tbl:tableE6", "ch07_unknown_and_mismatch_channels.tex",
     r"Unknown ISI $\to$ AWGN & AF & 0.1813",
     r"Unknown ISI $\to$ AWGN & AF & 0.9999"),
    ("arch:relay-param-counts", "ch07_unknown_and_mismatch_channels.tex",
     r"$11 \to 13 \to 1$, which is $170$ parameters",
     r"$11 \to 13 \to 1$, which is $169$ parameters"),
    ("tbl:slicer-floor-inline", "ch07_unknown_and_mismatch_channels.tex",
     r"DF closed form, Eq.~\eqref{eq:slicer-floor} & $0.1786$",
     r"DF closed form, Eq.~\eqref{eq:slicer-floor} & $0.9999$"),
    ("prose:mmse-monotonicity", "ch07_unknown_and_mismatch_channels.tex",
     r"($0.0666, 0.0374, 0.0354, 0.0333$ at $16$~dB",
     r"($0.9999, 0.0374, 0.0354, 0.0333$ at $16$~dB"),
    ("tbl:mmse-baseline", "ch07_unknown_and_mismatch_channels.tex",
     r"ISI (BPSK)      & $+6.58$",
     r"ISI (BPSK)      & $+9.99$"),
    ("tbl:table44", "ch05_experiments.tex",
     r"20 & 0.001378 & \textbf{0.001064} & 0.001092",
     r"20 & 0.001378 & \textbf{0.001064} & 0.009999"),
]


def _run(tex_dir):
    r = subprocess.run(
        [sys.executable, str(ROOT / "verify_thesis_tables.py"), "--tex", str(tex_dir)],
        capture_output=True, text=True, cwd=ROOT, timeout=900)
    return r.returncode, r.stdout + r.stderr


@pytest.fixture
def scratch_tex(tmp_path):
    dest = tmp_path / "chapters"
    shutil.copytree(CHAPTERS, dest)
    return dest


def test_clean_tree_passes(scratch_tex):
    """The control. Without it a mutation test proves nothing: a verifier that
    always fails would pass every case below."""
    code, out = _run(scratch_tex)
    assert code == 0, f"unmutated thesis should verify clean:\n{out[-3000:]}"
    assert "inconsistencies: 0" in out


@pytest.mark.parametrize("check,fname,old,new", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_is_caught(scratch_tex, check, fname, old, new):
    """Perturb one published number; the owning check must flag it."""
    path = scratch_tex / fname
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (
        f"anchor for {check} no longer appears exactly once in {fname}; "
        "the thesis moved and this mutation needs re-aiming")
    path.write_text(text.replace(old, new), encoding="utf-8")

    code, out = _run(scratch_tex)
    assert code != 0, f"{check} did not fail on a mutated value:\n{out[-3000:]}"
    assert f"[{check}]" in out, (
        f"something failed, but not {check} -- the mutation may be caught by "
        f"the wrong check:\n{out[-3000:]}")


def test_proof_copy_drift_is_caught(tmp_path):
    """The thesis and README state the same six proofs; a fix to one copy that
    misses the other has slipped through twice. Perturbing the README's copy
    while the thesis stays correct must be caught."""
    readme = ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    i = original.index("Proof of Claims")
    mutated = original[:i] + original[i:].replace("0.152", "0.999", 1)
    assert mutated != original, "README proof-of-claims anchor moved"
    try:
        readme.write_text(mutated, encoding="utf-8")
        code, out = _run(CHAPTERS)
        assert code != 0 and "[consistency:proof-copies]" in out, (
            f"drift between the two proof copies was not caught:\n{out[-2000:]}")
    finally:
        readme.write_text(original, encoding="utf-8")


def test_every_check_has_a_coverage_floor():
    """A check with no floor can silently drop to zero cells and still pass."""
    sys.path.insert(0, str(ROOT))
    import verify_thesis_tables as v
    code, out = _run(CHAPTERS)
    ran = set(re.findall(r"^((?:tbl|prose|arch):[\w:-]+)\s+\d+", out, re.M))
    missing = sorted(ran - set(v.MIN_CELLS))
    assert not missing, f"checks without a MIN_CELLS floor: {missing}"
