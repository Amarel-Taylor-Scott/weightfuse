"""weightfuse CLI."""

from __future__ import annotations

import argparse

from .synth import make_clean, make_mixed_problem
from .validate import held_out


def _verdict(r):
    return "REAL - fused weights beat single signals + null" if r["beats_null"] else "no fused gain"


def _show(r):
    print("  selected mode     : %s" % r["best_mode"])
    print("  held-out transfer : %.4f   (uniform %.4f, lift %+.4f)" %
          (r["learned"], r["uniform"], r["lift"]))
    print("  best single / gap : %.4f / %+.4f" % (r["best_single"], r["single_gap"]))
    print("  null p95 / p      : %.4f / %.3f" % (r["null_p95"], r["null_p"]))
    print("  VERDICT           : %s" % _verdict(r))


def cmd_demo(_args) -> int:
    print("weightfuse combines recency and residual-stability weights, then")
    print("requires the fused vector to beat uniform, best single signal, and null.\n")
    print("## 1. Drift plus corrupted recent rows")
    X, y, _ = make_mixed_problem(seed=0)
    _show(held_out(X, y, seed=0))
    print("\n## 2. Clean control (must NOT claim a fused gain)")
    X, y, _ = make_clean(seed=0)
    _show(held_out(X, y, seed=0))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="weightfuse", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
