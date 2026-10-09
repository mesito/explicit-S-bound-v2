# explicit-S-bound-v2

Companion repository for

> M. Ismail, *Explicit bounds for S(T) at finite heights* (2026).

Nine explicit bounds |S(T)| ≤ C₁ log T + C₂ log log T + C₃, valid for all T ≥ e, obtained in the framework of
Bellotti–Wong (Math. Comp. 2025) with parameters selected at finite heights, the Hiary–Patel–Yang input on the
critical line for the rows selected at heights up to 10³⁰ (the Patel–Yang sub-Weyl input beyond), and a 1-line input
c₁ log|Q₀+1+it| with c₁ = 0.532008 (the estimate ½ log t + 0.6633 of Hoo–Teo absorbed into the shift Q₀ = 10⁹).
Headline row: |S(T)| ≤ 0.134791 log T + 0.159606 log log T + 1.505817.
Every constant is a certified upper bound of an enclosure computed in Arb ball arithmetic (200 bits, rigorous
quadrature), rounded up on the exact upper endpoint in integer arithmetic.

```
sbound.py             constants of Bellotti–Wong Thm 1.2/1.4 (mpmath, tanh–sinh quadrature)
sbound_arb.py         the same in Arb ball arithmetic (python-flint), rigorous quadrature at 200 bits
verify_S_bound.py     52 PASS/FAIL checks (see below)
runlog_S_bound.txt    output of a run: PASS = 52, FAIL = 0
rows_theorem_1_1.json, rows_certified.json   the nine rows (raw, rounded, certified)
optimize_rows.py      the parameter selection of Section 4 (two chains, cross-checks, scans over Q₀ and n)
                      and the intermediate configurations of Remark 5.1
optimized_rows.json, optimize_rows_log.txt   output of optimize_rows.py (reproduces the parameters of Table 1)
```

## Run

    pip install mpmath numpy scipy python-flint
    python3 verify_S_bound.py            # the proof-relevant checks, under a minute
    python3 optimize_rows.py             # provenance of the parameters (all five stages)

`optimize_rows.py` runs in five stages of a few minutes each, which can also be run separately and resume after an
interruption: `python3 optimize_rows.py chain_hpy | chain_sw | cross | scans | diagnostics`, then
`python3 optimize_rows.py merge` (which writes `optimized_rows.json` only when all stages are complete).
The proof does not depend on it.

Tested with Python 3.12.3, python-flint 0.9.0, mpmath 1.3.0, numpy 2.4.4, scipy 1.17.1.
`Params` in `sbound.py` / `sbound_arb.py` defaults to the preset `PRESET_HPY` (shifts Q₁ = Q₂ = 1.5,
Q₃ = Q₁₀ = Q₁₁ = 10⁹); the Bellotti–Wong shifts are available as `PRESET_SUBWEYL_BW` and are admissible only for
the sub-Weyl input.

Groups of `verify_S_bound.py`: G1 reproduction of Bellotti–Wong Table 2 (five rows, their inputs and shifts);
G2 the 1-line input lemma (deterministic checks of the elementary constants and of the right sides, grid sanity
checks of the shifted inequalities); G3 the nine rows at 30 digits with admissibility, rounding-up and the
finite-height terms κ₃; G4 coverage of e ≤ T ≤ 3.06×10¹⁰, corollaries; G5 certification of the nine rows in ball
arithmetic and a rigorous proof of M(T) < B(T) in Proposition 1.2 (endpoints and critical point of every difference);
G6 certification of the shifted critical-line inequality with Q₁ = 1.5 on |t| ≤ 3 (t as a ball) and the failure of
the Bellotti–Wong shift 1.18 for the Hiary–Patel–Yang input; G7 the gains in Proposition 1.2 (certified bounds
0.508% and 8.965% on the whole interval by adaptive bisection in ball arithmetic, enclosures at the two extremal
points, the point gains); G8 the values 5.678, 5.357, 5.295, 5.256 of Remark 5.1.
