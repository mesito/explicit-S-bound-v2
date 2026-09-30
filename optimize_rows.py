#!/usr/bin/env python3
"""optimize_rows.py -- provenance of the parameter selection of Section 4 of the paper.

Objective at a target height T*:  C1 log T* + min{ C2 loglog T* + C3~ , C2' loglog T* + C3~' },
minimised over (c, r, eta) subject to the admissibility conditions (2.1), by Nelder-Mead on the mpmath
evaluation of sbound.py at 15 digits (xatol 1e-4, fatol 1e-5, at most 200 iterations per start).

It reproduces, in this order:
  1. the two chains of Section 4 (n = 4, Q0 = 1e9, c1 = c1(Q0) of Lemma 3.2, shifts of Remark 3.3):
     Hiary-Patel-Yang input for T* = T0, 3e12, 1e15, 1e20, 1e24, 1e30; sub-Weyl input for 1e40, 1e60, 1e100.
     The first target of a chain starts from rows 2, 3, 5 of Bellotti-Wong, Table 2; every later target from
     the optimum at the previous target and from row 3 of that table;
  2. the cross-checks of the regime choice: the other input at 1e30 and at 1e40 (three starts each);
  3. the scans at T* = 3e12 behind the choices n = 4 and Q0 = 1e9: Q0 in {1e8, 1e10} with n = 4, and
     n in {5, 6, 7} with Q0 = 1e9 (the case n = 4, Q0 = 1e9 is row 2 of the chain);
  4. the two intermediate configurations of Remark 5.1 at T* = T0: the inputs and shifts of Bellotti-Wong
     (sub-Weyl, c1 = 1, n = 5), and the Hiary-Patel-Yang input with c1 = 1, n = 5 and the shifts
     Q0 = 1, Q1 = Q2 = 1.5, Q3 = Q11 = 6, Q10 = 2.3 (admissible for that input and for c1 = 1).
The minimiser lies on the constraint 1 + eta < c; parameters are then "cleaned" as in Section 4:
c = 1 + eta + 1e-6 (rounded to 7 decimals), r rounded to 6 decimals, r = 2c - 1 + 1e-6 where c - r < 1 - c is
active and r = c + 1/2 - 1e-6 where c - r > -1/2 is active (the latter only in the first configuration of item 4).

The proof does not depend on this script: verify_S_bound.py certifies the published rows as they stand.
Usage:  python3 optimize_rows.py [chain_hpy | chain_sw | cross | scans | diagnostics | merge]   (default: all stages)
"""
import math, json, os
from scipy.optimize import minimize
from sbound import Params, HPY, SUBWEYL, c1_of_Q0

T0 = 30610046000
STARTS = [(1.070007, 1.182997, 0.069901), (1.043400, 1.250450, 0.040000), (1.499159, 1.998357, 0.499050)]


def cfg_main(k, Q0=1e9, n=4):
    """configuration of Table 1: Lemma 3.2 input with shift Q0, critical-line shifts of Remark 3.3"""
    return dict(k=k, n=n, c1=c1_of_Q0(Q0), c2=1, Q0=Q0, Q1=1.5, Q2=1.5, Q3=Q0, Q10=Q0, Q11=Q0, Qsig=1)


CFG_BW = dict(k=SUBWEYL, n=5, c1=1, c2=1, Q0=1, Q1=1.18, Q2=1.18, Q3=3.9, Q10=2.3, Q11=3.9, Qsig=1)
CFG_HPY_C1 = dict(k=HPY, n=5, c1=1, c2=1, Q0=1, Q1=1.5, Q2=1.5, Q3=6, Q10=2.3, Q11=6, Qsig=1)


def value(o, T):
    L = math.log(T); l = math.log(L)
    return min(float(o['C1']) * L + float(o['C2']) * l + float(o['C3T']), float(o['C1']) * L + float(o['C2p']) * l + float(o['C3Tp']))


def objective(x, T, cfg):
    c, r, eta = x
    try:
        P = Params(c, r, eta, dps=15, **cfg)
        if not P.admissible():
            return 1e3
        return value(P.constants(), T)
    except Exception:
        return 1e3


def optimise_from(T, cfg, starts):
    best = None
    for s0 in starts:
        res = minimize(objective, s0, args=(T, cfg), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, maxiter=200))
        if best is None or res.fun < best.fun:
            best = res
    return best


def clean(c, r, eta):
    """c = 1 + eta + 1e-6; r rounded to 6 decimals, then kept strictly inside the two constraints on r
    that can be active: r > 2c - 1 (i.e. c - r < 1 - c) and r < c + 1/2 (i.e. c - r > -1/2)."""
    eta = round(eta, 6); c = round(1 + eta + 1e-6, 7); r = round(r, 6)
    if r - (2 * c - 1) < 1e-4:
        r = math.ceil(round((2 * c - 1 + 1e-6) * 1e6, 3)) / 1e6
    if (c + 0.5) - r < 1e-4:
        r = math.floor(round((c + 0.5 - 1e-6) * 1e6, 3)) / 1e6
    return c, r, eta


def report(label, T, cfg, b):
    c, r, eta = clean(*b.x)
    P = Params(c, r, eta, dps=15, **cfg); ok = P.admissible()
    v = value(P.constants(), T) if ok else float('nan')
    print("  %-26s T*=%-10.4g minimum %.6f  cleaned (c, r, eta) = (%.7f, %.6f, %.6f)  admissible=%s  value %.6f"
          % (label, T, b.fun, c, r, eta, ok, v), flush=True)
    return dict(label=label, T=T, c=c, r=r, eta=eta, admissible=ok, value=v, raw_minimum=float(b.fun), x=[float(t) for t in b.x])


def load(name):
    f = "optimized_rows_%s.json" % name
    return json.load(open(f)) if os.path.exists(f) else []


def save(name, out):
    json.dump(out, open("optimized_rows_%s.json" % name, "w"), indent=1)


def stage_chain(regime):
    name = "chain_" + regime.lower()
    k, targets = (HPY, [T0, 3e12, 1e15, 1e20, 1e24, 1e30]) if regime == 'HPY' else (SUBWEYL, [1e40, 1e60, 1e100])
    out = load(name)
    for T in targets[len(out):]:
        starts = [tuple(out[-1]['x']), STARTS[1]] if out else STARTS
        b = optimise_from(T, cfg_main(k), starts)
        d = report("row, %s" % regime, T, cfg_main(k), b); d['regime'] = regime; out.append(d); save(name, out)
    return out


def stage_cross():
    out = load("cross")
    for T, regime, k in ((1e30, 'SW', SUBWEYL), (1e40, 'HPY', HPY))[len(out):]:
        d = report("cross-check, %s" % regime, T, cfg_main(k), optimise_from(T, cfg_main(k), STARTS)); d['regime'] = regime
        out.append(d); save("cross", out)
    return out


def stage_scans():
    out = load("scans")
    for Q0, n in ((1e8, 4), (1e10, 4), (1e9, 5), (1e9, 6), (1e9, 7))[len(out):]:
        d = report("scan, Q0=%.0e n=%d" % (Q0, n), 3e12, cfg_main(HPY, Q0=Q0, n=n), optimise_from(3e12, cfg_main(HPY, Q0=Q0, n=n), STARTS))
        d['Q0'] = Q0; d['n'] = n; out.append(d); save("scans", out)
    return out


def stage_diagnostics():
    out = load("diagnostics")
    for name, cfg in (("Bellotti-Wong inputs", CFG_BW), ("HPY input, c1 = 1", CFG_HPY_C1))[len(out):]:
        out.append(report(name, T0, cfg, optimise_from(T0, cfg, STARTS))); save("diagnostics", out)
    return out


STAGES = [("chain_hpy", "1a. chain, Hiary-Patel-Yang input (rows 1-6)", lambda: stage_chain('HPY')),
          ("chain_sw", "1b. chain, sub-Weyl input (rows 7-9)", lambda: stage_chain('SW')),
          ("cross", "2. cross-checks of the regime choice", stage_cross),
          ("scans", "3. scans at T* = 3e12 (Hiary-Patel-Yang input)", stage_scans),
          ("diagnostics", "4. intermediate configurations of Remark 5.1 at T0", stage_diagnostics)]


if __name__ == "__main__":
    # python3 optimize_rows.py            -> all stages in sequence
    # python3 optimize_rows.py <stage>    -> one stage (chain_hpy, chain_sw, cross, scans, diagnostics); a stage
    #                                        resumes from its file optimized_rows_<stage>.json if interrupted
    # python3 optimize_rows.py merge      -> combine the stage files into optimized_rows.json and print them
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, title, fn in STAGES:
        if which in ("all", name):
            print(title, flush=True); fn()
    if which in ("all", "merge"):
        merged = {name: load(name) for name, _, _ in STAGES}
        size = dict(chain_hpy=6, chain_sw=3, cross=2, scans=5, diagnostics=2)
        missing = [name for name in size if len(merged[name]) != size[name]]
        if missing:
            sys.exit("incomplete stages %s: run  python3 optimize_rows.py <stage>  first (nothing written)" % missing)
        json.dump(merged, open("optimized_rows.json", "w"), indent=1)
        for name, title, _ in STAGES:
            print(title)
            for d in merged[name]:
                print("  %-26s T*=%-10.4g minimum %.6f  cleaned (c, r, eta) = (%.7f, %.6f, %.6f)  admissible=%s  value %.6f"
                      % (d['label'], d['T'], d['raw_minimum'], d['c'], d['r'], d['eta'], d['admissible'], d['value']))
