# explicit-S-bound-v2

Companion repository for

> M. Ismail, *Explicit bounds for S(T) at finite heights* (2026).

Nine explicit bounds |S(T)| ≤ C₁ log T + C₂ log log T + C₃, valid for all T ≥ e, obtained in the framework of
Bellotti–Wong (Math. Comp. 2025) with parameters selected at finite heights, the Hiary–Patel–Yang input on the
critical line for the rows selected at heights up to 10³⁰ (the Patel–Yang sub-Weyl input beyond), and a 1-line input c₁ log|Q₀+1+it| with c₁ = 0.532008 (the estimate ½ log t + 0.6633 of Hoo–Teo absorbed into the
shift Q₀ = 10⁹). Headline row: |S(T)| ≤ 0.134791 log T + 0.159606 log log T + 1.505817.
Every constant is a certified upper bound of an enclosure computed in Arb ball arithmetic (200 bits, rigorous quadrature),
rounded up on the exact upper endpoint in integer arithmetic.

```
sbound.py             constants of Bellotti–Wong Thm 1.2/1.4 (mpmath, tanh–sinh quadrature)
sbound_arb.py         the same in Arb ball arithmetic (python-flint), rigorous quadrature at 200 bits
verify_S_bound.py     42 PASS/FAIL checks (see below)
runlog_S_bound.txt    output of a run: PASS = 42, FAIL = 0
rows_theorem_1_1.json, rows_certified.json   the nine rows (raw, rounded, certified)
```

## Run

    pip install mpmath numpy python-flint
    python3 verify_S_bound.py

Groups: G1 reproduction of Bellotti–Wong Table 2 (five rows, their inputs); G2 the 1-line input lemma
(deterministic checks of the elementary constants and of the right sides, grid sanity checks of the shifted inequalities); G3 the nine rows at 30 digits with admissibility, rounding-up and
the finite-height terms κ₃; G4 coverage of e ≤ T ≤ 3.06×10¹⁰, corollaries; G5 certification of the nine rows
in ball arithmetic and a rigorous proof of Proposition 1.2 (endpoints and critical point of every difference);
G6 certification of the shifted critical-line inequality with Q₁ = 1.5 on |t| ≤ 3 (t as a ball) and the failure of the
Bellotti–Wong shift 1.18 for the Hiary–Patel–Yang input.
