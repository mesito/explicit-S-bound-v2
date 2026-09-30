#!/usr/bin/env python3
"""optimize_rows.py -- provenance of the parameters of Table 1 (Section 4 of the paper).

For a target height T* it minimises, over (c, r, eta) subject to the admissibility conditions (2.1),
    C1 log T* + min{ C2 loglog T* + C3~ , C2' loglog T* + C3~' }
with n = 4, Q0 = 1e9, c1 = c1(Q0) (Hoo-Teo), and the shifts of Remark 3.3 (Q1 = Q2 = 1.5, Q3 = Q10 = Q11 = Q0,
Yang's lines 1), by Nelder-Mead on the mpmath evaluation of sbound.py at 15 digits.  The targets are run in two
chains, as in the paper: the Hiary-Patel-Yang input for T0, 3e12, 1e15, 1e20, 1e24, 1e30 and the sub-Weyl input
for 1e40, 1e60, 1e100.  The first target of a chain is started from rows 2, 3 and 5 of Bellotti-Wong, Table 2;
each later target from the optimum of the previous target and from row 3 of that table.  The regime choice is
cross-checked by optimising the other input at the two boundary targets 1e30 and 1e40 (three starts each).
The minimiser lies on the constraint 1 + eta < c; the published parameters are then "cleaned" as in Section 4:
c = 1 + eta + 1e-6 (rounded to 7 decimals), r rounded to 6 decimals, and r = 2c - 1 + 1e-6 where the constraint
c - r < 1 - c is active.

The proof does not depend on this script: verify_S_bound.py certifies the published rows as they stand.
Runtime: about 15 minutes.

Usage:  python3 optimize_rows.py
"""
import sys, math, json
from scipy.optimize import minimize
from sbound import Params, HPY, SUBWEYL, c1_of_Q0

T0 = 30610046000
Q0 = 1e9
STARTS = [(1.070007, 1.182997, 0.069901), (1.043400, 1.250450, 0.040000), (1.499159, 1.998357, 0.499050)]
SHIFTS = {'HPY': dict(k=HPY, Q1=1.5, Q2=1.5), 'SW': dict(k=SUBWEYL, Q1=1.5, Q2=1.5)}   # the shifts of Remark 3.3 (verify_S_bound.py, G3/G5/G6)
C1 = c1_of_Q0(Q0)


def params(c, r, eta, regime, dps=15):
    return Params(c, r, eta, n=4, c1=C1, c2=1, Q0=Q0, Q3=Q0, Q10=Q0, Q11=Q0, Qsig=1, dps=dps, **SHIFTS[regime])


def value(o, T):
    L = math.log(T); l = math.log(L)
    return min(float(o['C1']) * L + float(o['C2']) * l + float(o['C3T']), float(o['C1']) * L + float(o['C2p']) * l + float(o['C3Tp']))


def objective(x, T, regime):
    c, r, eta = x
    try:
        P = params(c, r, eta, regime)
        if not P.admissible():
            return 1e3
        return value(P.constants(), T)
    except Exception:
        return 1e3



def clean(c, r, eta):
    eta = round(eta, 6); c = round(1 + eta + 1e-6, 7); r = round(r, 6)
    if r - (2 * c - 1) < 1e-4:
        r = round(2 * c - 1 + 1e-6, 6)
    return c, r, eta


CHAINS = [('HPY', [T0, 3e12, 1e15, 1e20, 1e24, 1e30]), ('SW', [1e40, 1e60, 1e100])]
CROSS = [(1e30, 'SW'), (1e40, 'HPY')]


def optimise_from(T, regime, starts):
    best = None
    for s0 in starts:
        res = minimize(objective, s0, args=(T, regime), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, maxiter=200))
        if best is None or res.fun < best.fun:
            best = res
    return best


def report(T, regime, b):
    c, r, eta = clean(*b.x)
    P = params(c, r, eta, regime); ok = P.admissible()
    v = value(P.constants(), T) if ok else float('nan')
    print("  T*=%-10.4g %-3s  minimum %.6f  cleaned (c, r, eta) = (%.7f, %.6f, %.6f)  admissible=%s  value %.6f"
          % (T, regime, b.fun, c, r, eta, ok, v), flush=True)
    return dict(T=T, regime=regime, c=c, r=r, eta=eta, admissible=ok, value=v, raw_minimum=float(b.fun))


if __name__ == "__main__":
    rows = []
    for regime, targets in CHAINS:
        starts = STARTS
        for T in targets:
            b = optimise_from(T, regime, starts)
            rows.append(report(T, regime, b))
            starts = [tuple(b.x), STARTS[1]]
    print("  cross-checks of the regime choice:", flush=True)
    cross = [report(T, regime, optimise_from(T, regime, STARTS)) for T, regime in CROSS]
    json.dump(dict(rows=rows, cross=cross), open("optimized_rows.json", "w"), indent=1)
