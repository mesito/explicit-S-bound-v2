# explicit-S-bound-v2

Companion repository for

> M. Ismail, *Explicit bounds for S(T) at finite heights* (2026).

Nine explicit bounds |S(T)| ≤ C₁ log T + C₂ log log T + C₃, valid for all T ≥ e, obtained in the framework of
Bellotti–Wong (Math. Comp. 2025) with parameters selected at finite heights, the Hiary–Patel–Yang input on the
critical line for T ≤ e¹⁰⁵, and a 1-line input c₁ log|Q₀+1+it| with c₁ = 0.571050 (Patel and Hiary–Leong–Yang
absorbed into the shift Q₀ = 10⁹). Headline row: |S(T)| ≤ 0.134854 log T + 0.158698 log log T + 1.511684.
Every constant is a certified upper bound of an enclosure computed in Arb ball arithmetic.

```
sbound.py             constants of Bellotti–Wong Thm 1.2/1.4 (mpmath, tanh–sinh quadrature)
sbound_arb.py         the same in Arb ball arithmetic (python-flint), rigorous quadrature at 200 bits
verify_S_bound.py     39 PASS/FAIL checks (see below)
runlog_S_bound.txt    output of a run: PASS = 39, FAIL = 0
rows_theorem_1_1.json, rows_certified.json   the nine rows (raw, rounded, certified)
```

## Run

    pip install mpmath numpy python-flint
    python3 verify_S_bound.py

Groups: G1 reproduction of Bellotti–Wong Table 2 (five rows, their inputs); G2 the 1-line input lemma
(analytic steps and grid sanity checks); G3 the nine rows at 30 digits with admissibility, rounding-up and
the finite-height terms κ₃; G4 coverage of e ≤ T ≤ 3.06×10¹⁰, corollaries; G5 certification of the nine rows
in ball arithmetic and a rigorous proof of Proposition 1.2 (endpoints and critical point of every difference).
