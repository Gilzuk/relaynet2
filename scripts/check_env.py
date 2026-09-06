#!/usr/bin/env python3
"""Environment doctor for the reproduction suite.

Reports what is installed, which reproduction tiers are therefore available,
and what is missing for the rest. Exit code 0 if at least the verification
tier can run, 1 otherwise.

Usage:  python3 scripts/check_env.py     (or: make check)
"""
import importlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, NO = "  OK  ", " MISS "


def probe(module):
    try:
        m = importlib.import_module(module)
        return True, getattr(m, "__version__", "?")
    except Exception:
        return False, None


def main():
    print("=" * 70)
    print("relaynet reproduction suite -- environment check")
    print("=" * 70)

    print(f"\npython {sys.version.split()[0]}  ({sys.executable})")

    print("\nRequired (verification + unknown-channel tiers):")
    required = {}
    for mod in ("numpy", "scipy", "matplotlib"):
        got, ver = probe(mod)
        required[mod] = got
        print(f"{OK if got else NO} {mod:<12} {ver or 'not installed'}")

    print("\nOptional:")
    got_torch, tver = probe("torch")
    print(f"{OK if got_torch else NO} {'torch':<12} "
          f"{tver or 'not installed (needed only for make repro-full)'}")
    if got_torch:
        import torch
        cuda = torch.cuda.is_available()
        print(f"{OK if cuda else '  --  '} {'cuda':<12} "
              f"{'available' if cuda else 'CPU only -- repro-full will be slow'}")
    got_pytest, pver = probe("pytest")
    print(f"{OK if got_pytest else NO} {'pytest':<12} "
          f"{pver or 'not installed (needed only for make test)'}")

    print("\nData and tooling:")
    checks = [
        ("results/", os.path.isdir(os.path.join(ROOT, "results"))),
        ("e6_unknown_channel_results/",
         os.path.isdir(os.path.join(ROOT, "e6_unknown_channel_results"))),
        ("verify_thesis_tables.py",
         os.path.isfile(os.path.join(ROOT, "verify_thesis_tables.py"))),
        ("thesis/chapters/", os.path.isdir(os.path.join(ROOT, "thesis", "chapters"))),
    ]
    for name, present in checks:
        print(f"{OK if present else NO} {name}")

    print(f"{OK if shutil.which('latexmk') else '  --  '} "
          f"latexmk       {'found' if shutil.which('latexmk') else 'not found (only needed for make thesis)'}")

    core = all(required.values())
    print("\n" + "-" * 70)
    print("Available reproduction tiers:")
    print(f"  make verify         {'YES' if core else 'NO  (install numpy/scipy)'}"
          "   -- check every thesis number against its data source (~1 min)")
    print(f"  make repro-unknown  {'YES' if core else 'NO  (install numpy/scipy)'}"
          "   -- re-run the unknown-channel study (~40 min, no torch)")
    print(f"  make repro-full     {'YES' if got_torch else 'NO  (install torch)'}"
          "   -- re-run every experiment (hours; GPU strongly advised)")
    print("-" * 70)

    if not core:
        print("\nInstall the core dependencies first:")
        print("  python3 -m pip install -r requirements-repro.txt")
        return 1
    print("\nReady. Start with:  make verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
