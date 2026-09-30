#!/usr/bin/env python3
"""verify_S_bound.py -- verification suite for
  M. Ismail, "Explicit bounds for S(T) at finite heights" (2026).

Requires: mpmath, numpy.  Uses sbound.py (independent mpmath implementation of the
constants of Bellotti-Wong, Math. Comp. (2025), Theorems 1.2/1.4).

Groups:
  G1  reproduction of Bellotti-Wong Table 2 (all five rows, sub-Weyl input, c1 = 1)
  G2  the 1-line input: c1(Q0) = 1/2 + 0.6633/log Q0 from Hoo-Teo (|zeta(1+it)| <= 1/2 log t + 0.6633, t >= e);
      unshifted form for t >= t0 = Q0, shifted forms for all real t (Lemma 3.2); the elementary constants
      of the proof evaluated deterministically, plus grid sanity checks for 0 < t <= 5e4
  G3  the nine rows of Theorem 1.1: admissibility, 30-digit evaluation, rounding-up,
      the finite-height terms kappa_3
  G4  coverage of e <= T <= 3.06e10 through computed bounds; comparison with every
      tabulated Bellotti-Wong bound; Corollaries (gap bound, N(T) form)
  G7  Proposition 1.2, the gains: certified lower bound 0.508% and upper bound 8.965% on [T0, 1e200] by adaptive
      bisection in ball arithmetic (monotonicity of M and B), enclosures at the extremal points, the point gains
  G8  Remark 5.1: the values 5.678, 5.357, 5.295, 5.256 at T0 (parameters from optimize_rows.py)
  G6  the shifted critical-line input with Q1 = 1.5: certified on |t| <= 3 in ball arithmetic (t as a ball),
      analytic for |t| >= 3; and the failure of the Bellotti-Wong shift 1.18 for the HPY input
  G5  CERTIFICATION (python-flint / Arb): every constant of Theorem 1.1 as a rigorous ball
      enclosure (200-bit ball arithmetic, acb_calc_integrate), the published constants being
      upper bounds of the enclosures rounded up; and a rigorous proof of Proposition 1.2
      (each difference alpha L + beta log L + gamma is checked at the segment endpoints and at
      its unique critical point, in ball arithmetic)
Every check prints PASS/FAIL.  Runtime under a minute.
"""
import sys, math, json, time
import mpmath as mp
import numpy as np
from sbound import Params, SUBWEYL, HPY, c1_of_Q0, PRESET_SUBWEYL_BW

RES = []
def check(name, ok, measured="", expected=""):
    RES.append(ok); print("  [%s] %-70s %s %s" % ("PASS" if ok else "FAIL", name, measured, ("(expected %s)" % expected) if expected else ""), flush=True)

T0 = 30610046000
Q0 = 1e9
Q1 = 1.5          # shift of the critical-line input (Lemma 3.1); Yang's lines keep Q4 = ... = Q8 = 1 as in Bellotti-Wong
BW_TABLE = [  # (c, r, eta) -> (C1, C2, C2', C3, C3', C3~, C3~')  [BW, Table 2]
    ((1.000225, 1.000605, 0.000158), (0.10076, 0.24460, 1.68845, 8.08344, 2.38456, 7.20844, 1.50956)),
    ((1.070007, 1.182997, 0.069901), (0.11000, 0.17447, 1.54543, 3.71067, 2.15392, 2.83567, 1.27892)),
    ((1.043400, 1.250450, 0.040000), (0.11200, 0.12567, 1.32678, 3.77417, 2.14783, 2.89916, 1.27283)),
    ((1.000060, 1.499556, 1.54244e-5), (0.12355, 0.06782, 0.97933, 6.25796, 2.05854, 5.38296, 1.18354)),
    ((1.499159, 1.998357, 0.499050), (0.16732, 0.17266, 1.61679, 1.96334, 1.40271, 1.08834, 0.52771)),
]
BW_S = [(0.10076, 0.24460, 7.20844), (0.10076, 1.68845, 1.50956), (0.11000, 0.17447, 2.83567), (0.11000, 1.54543, 1.27892),
        (0.11200, 0.12567, 2.89916), (0.11200, 1.32678, 1.27283), (0.12355, 0.06782, 5.38296), (0.12355, 0.97933, 1.18354),
        (0.16732, 0.17266, 1.08834), (0.16732, 1.61679, 0.52771)]
# Theorem 1.1: (T_opt, regime, c, r, eta); n = 4, Q0 = 1e9, c1 = c1(Q0) from Hoo-Teo throughout
ROWS = [(T0, 'HPY', 1.1948780, 1.667319, 0.194877),
        (3e+12, 'HPY', 1.1688080, 1.581471, 0.168807),
        (1e+15, 'HPY', 1.1450330, 1.501535, 0.145032),
        (1e+20, 'HPY', 1.1150630, 1.395716, 0.115062),
        (1e+24, 'HPY', 1.0994560, 1.339248, 0.099455),
        (1e+30, 'HPY', 1.0832160, 1.279677, 0.083215),
        (1e+40, 'SW', 1.0660270, 1.215723, 0.066026),
        (1e+60, 'SW', 1.0677880, 1.135577, 0.067787),
        (1e+100, 'SW', 1.0428930, 1.085787, 0.042892)]
f = lambda p, T: p[0] * math.log(T) + p[1] * math.log(math.log(T)) + p[2]

def G1():
    print("\nG1  Bellotti-Wong Table 2 with their inputs and shifts: sub-Weyl 66.7 t^{27/164}, c1 = 1, n = 5,\n    (Q0,Q1,Q2,Q3,Q4..Q8,Q10,Q11) = (1,1.18,1.18,3.9,1..1,2.3,3.9) = PRESET_SUBWEYL_BW")
    for (c, r, e), ref in BW_TABLE:
        P = Params(c, r, e, n=5, c1=1, c2=1, Q0=1, Qsig=1, dps=20, **PRESET_SUBWEYL_BW); o = P.constants()
        assert (float(P.Q1), float(P.Q2), float(P.Q3), float(P.Q10), float(P.Q11)) == (1.18, 1.18, 3.9, 2.3, 3.9)
        got = [float(o[k]) for k in ('C1', 'C2', 'C2p', 'C3', 'C3p', 'C3T', 'C3Tp')]
        dev = max(abs(g - x) for g, x in zip(got, ref))
        check("row (c,r,eta)=(%.6g,%.6g,%.3g) reproduced" % (c, r, e), dev < 1.5e-5 and P.admissible(), "max dev %.1e" % dev, "< 1.5e-5")

def G2():
    print("\nG2  the 1-line input c1(Q0) = 1/2 + 0.6633/log Q0 (Hoo-Teo: |zeta(1+it)| <= 1/2 log t + 0.6633, t >= e)")
    c1 = c1_of_Q0(Q0)
    check("c1(1e9) = 0.532008 (rounded up)", abs(float(c1) - 0.5320077) < 2e-6, "%.7f" % c1)
    # analytic steps used in the proof: for 3 <= t <= Q0, 1/2 log t + 0.6633 <= 1/2 log Q0 + 0.6633 = c1 log Q0; for t >= Q0,
    # (1/2 + 0.6633/log t) log t <= c1 log t.  Sanity checks of the two shifted inequalities on a grid (the proof for |t| <= 3 is analytic):
    mp.mp.dps = 15
    grid = list(np.linspace(0.01, 60, 120)) + list(np.geomspace(60, 5e4, 80))
    w1 = max(abs(mp.mpc(0, t) * mp.zeta(1 + 1j * mp.mpf(t))) / (float(c1) * abs(mp.mpc(Q0 + 1, t)) * mp.log(abs(mp.mpc(Q0 + 1, t)))) for t in grid)
    w0 = max(abs(mp.zeta(1j * mp.mpf(t))) / (float(c1) / math.sqrt(2 * math.pi) * abs(mp.mpc(Q0, t))**0.5 * mp.log(abs(mp.mpc(Q0, t)))) for t in grid)
    check("grid: |it zeta(1+it)| <= c1 |Q0+1+it| log|Q0+1+it| on 0<t<=5e4", w1 < 1, "max ratio %.1e" % w1)
    check("grid: |zeta(it)| <= c1 (2pi)^{-1/2} |Q0+it|^{1/2} log|Q0+it| on 0<t<=5e4", w0 < 1, "max ratio %.1e" % w0)
    # the two elementary bounds of the proof for 0<|t|<=e, checked deterministically (mpmath, 30 digits)
    mp.mp.dps = 30; e_ = mp.e
    b1 = mp.sqrt(1 + e_**2) * (1 + e_); b2 = mp.sinh(mp.pi * e_ / 2) / e_ * mp.sqrt(1 + e_**2) * (1 + e_) / mp.pi
    check("elementary bounds: sqrt(1+e^2)(1+e) < 11 and pi^-1 sinh(pi e/2)/e sqrt(1+e^2)(1+e) < 46", b1 < 11 and b2 < 46, "%.4f, %.4f" % (b1, b2))
    check("right sides for |t| <= e exceed 1e10 and 1e5", float(c1) * Q0 * math.log(Q0) > 1e10 and float(c1) / math.sqrt(2 * math.pi) * math.sqrt(Q0) * math.log(Q0) > 1e5, "%.2e, %.2e" % (float(c1) * Q0 * math.log(Q0), float(c1) / math.sqrt(2 * math.pi) * math.sqrt(Q0) * math.log(Q0)))
    mp.mp.dps = 15

def G3():
    print("\nG3  the nine rows of Theorem 1.1 (n = 4, Q0 = 1e9, 30 digits, rounded up at 6 decimals)")
    out = []
    for Topt, reg, c, r, e in ROWS:
        P = Params(c, r, e, n=4, k=(HPY if reg == 'HPY' else SUBWEYL), c1=c1_of_Q0(Q0), c2=1, Q0=Q0, Q1=Q1, Q2=Q1, Q3=Q0, Q10=Q0, Q11=Q0, dps=30)
        adm = P.admissible(); o = P.constants()
        up = lambda x: math.ceil(float(x) * 1e6) / 1e6
        row = dict(Topt=Topt, regime=reg, c=c, r=r, eta=e, C1=up(o['C1']), C2=up(o['C2']), C3T=up(o['C3T']), C3N=up(o['C3']),
                   raw=[float(mp.nstr(o[k], 12)) for k in ('C1', 'C2', 'C3T', 'C3')], kap3=float(o['kap3']), delta=float(P.delta))
        out.append(row)
        check("opt@%.0e %-3s admissible; (C1,C2,C3~) = (%.6f, %.6f, %.6f)" % (Topt, reg, row['C1'], row['C2'], row['C3T']), adm and row['kap3'] < 1e-3, "kappa3=%.1e delta=%.4f" % (row['kap3'], row['delta']))
    json.dump(out, open("rows_theorem_1_1.json", "w"), indent=1)
    return out

def G4(rows):
    print("\nG4  coverage, comparisons, corollaries")
    R = [(w['C1'], w['C2'], w['C3T']) for w in rows]
    cov = all(f(p, math.e) >= 1 and f(p, 280) >= 2 and f(p, 6.8e6) >= 2.5167 for p in R)
    check("every row >= 1 on [e,280], >= 2 on [280, 6.8e6], >= 2.5167 on [6.8e6, 3.06e10]", cov, "values at e: %s" % [round(f(p, math.e), 2) for p in R])
    worst = 1.0
    for T in np.geomspace(T0, 1e200, 400):
        worst = min(worst, 1 - min(f(p, T) for p in R) / min(f(p, T) for p in BW_S))
    check("min over rows below every tabulated Bellotti-Wong bound for T0 <= T <= 1e200 (grid)", worst > 0, "min relative gain %.3f%%" % (100 * worst))
    for T, ref in ((T0, 5.2563), (3e12, 5.9143), (1e15, 6.7108), (1e20, 8.2075), (1e30, 11.0118), (1e100, 28.5727)):
        v = min(f(p, T) for p in R); check("min over rows at T = %.0e" % T, abs(v - ref) < 2e-3, "%.4f" % v, "%.4f" % ref)
    S0 = min(f(p, 3e12) for p in R)
    check("gap corollary at 3e12: s_n <= 1 + 2 Sbar = 12.829", abs(1 + 2 * S0 - 12.829) < 1e-3, "%.3f" % (1 + 2 * S0))
    # N(T) form: C3 (N-form) = C3~ + 7/8 + 1/(50 T0) - (arctan terms)/pi ; check consistency per row
    ok = all(abs(w['C3N'] - (w['C3T'] + 7 / 8)) < 1e-4 for w in rows)
    check("N(T)-form constants = S-form constants + 7/8 (to 1e-4)", ok)

def G5(rows):
    print("\nG5  certification with Arb ball arithmetic (python-flint)")
    from flint import arb, ctx
    from sbound_arb import Params as ParamsA, SUBWEYL as SWA, HPY as HPYA, c1_of_Q0 as c1A
    ctx.prec = 200
    cert = []
    for (Topt, reg, c, r, e), w in zip(ROWS, rows):
        P = ParamsA(c, r, e, n=4, k=(HPYA if reg == 'HPY' else SWA), c1=c1A(10**9), c2=1, Q0=10**9, Q1='1.5', Q2='1.5', Q3=10**9, Q10=10**9, Q11=10**9, prec=200)
        o = P.constants(); ups = {}
        for k, key in (('C1', 'C1'), ('C2', 'C2'), ('C3T', 'C3T'), ('C3', 'C3N')):
            # exact directed rounding: upper endpoint of the ball (exact), times 10^6, ceiling as an exact integer
            up = (o[k].upper() * 1000000).ceil().unique_fmpz()
            assert up is not None
            ups[key] = int(up) / 1e6
        ok = P.admissible() and all(ups[key] == w[key] for key in ('C1', 'C2', 'C3T', 'C3N')) and max(float(o[k].rad()) for k in ('C1', 'C2', 'C3T')) < 1e-8
        check("opt@%.0e certified enclosures; rounded upper bounds equal the published constants" % Topt, ok, "radii <= %.1e" % max(float(o[k].rad()) for k in ('C1', 'C2', 'C3T')))
        cert.append(ups)
    ctx.prec = 200
    R = [(arb(str(w['C1'])), arb(str(w['C2'])), arb(str(w['C3T']))) for w in cert]
    B = [(arb(str(a)), arb(str(b)), arb(str(cc))) for a, b, cc in BW_S]
    L1a, L2a = arb(T0).log(), 200 * arb(10).log()          # exact endpoints: T0 = 30 610 046 000 and 10^200
    L1, L2 = float(L1a.mid()), float(L2a.mid())
    # segments with the decimal breakpoints stated in the paper, used as exact endpoints (rows 1..9 in order)
    BP = ['26.33', '31.56', '39.86', '50.44', '61.68', '81.87', '97.58', '144.07']
    ends = [L1a] + [arb(b) for b in BP] + [L2a]
    segs = [(ends[i], ends[i + 1], i) for i in range(len(R))]
    allok = True; worst = arb(-1e9); wpair = None
    fm = lambda x: float(x.mid()) if isinstance(x, arb) else float(x)
    print("      segments (L-range -> row): " + "; ".join("[%.2f, %.2f] -> %d" % (fm(a), fm(b), i + 1) for a, b, i in segs))
    for a, b, i in segs:
        a, b = (a if isinstance(a, arb) else arb(a)), (b if isinstance(b, arb) else arb(b))
        for jq, q in enumerate(B):
            al, be, ga = R[i][0] - q[0], R[i][1] - q[1], R[i][2] - q[2]
            D = lambda L: al * L + be * L.log() + ga
            vals = [D(a), D(b)]; Ls = -be / al
            if (Ls > a) and (Ls < b): vals.append(D(Ls))
            for v in vals:
                if not (v < 0): allok = False
                if v > worst: worst = v; wpair = (i + 1, jq + 1)
    check("Proposition 1.2 proved in ball arithmetic: M(T) < B(T) on [T0, 1e200], T0 = 30610046000 (%d segments)" % len(segs), allok, "max difference %s (row %d vs BW bound %d)" % (worst.str(5), wpair[0], wpair[1]), "< 0")

def G6():
    print("\nG6  shifted critical-line input (Lemma 3.1): |(s-1) zeta(s)| <= k1 |Q1+s|^{k2+1} (log|Q1+s|)^{k3}, s = 1/2+it, Q1 = 1.5")
    from flint import arb, acb, ctx
    ctx.prec = 100
    # |t| <= 3 certified with t as a ball on 600 subintervals (HPY: k1 = 0.618, k2 = 1/6, k3 = 1); for |t| >= 3 the
    # inequality follows from |zeta(1/2+it)| <= 0.618 t^{1/6} log t, |s-1| <= |Q1+s| and t <= |Q1+s|
    Q = arb('1.5'); k1 = arb('0.618'); N = 600; h = arb(3) / N; worst = None; ok = True
    for i in range(N):
        tb = (h * i).union(h * (i + 1)); s_ = acb(arb('0.5'), tb); m = abs(acb(Q + arb('0.5'), tb))
        gap = k1 * m**(arb(7) / 6) * m.log() - abs((s_ - 1) * s_.zeta())
        if not (gap > 0): ok = False
        if worst is None or gap.lower() < worst: worst = gap.lower()
    check("HPY shifted inequality certified on 0 <= t <= 3 (ball arithmetic, %d subintervals)" % N, ok, "minimal margin %s" % worst.str(4), "> 0")
    # sub-Weyl input (rows 7-9): on |t| <= 3, |(s-1) zeta(s)| <= 3.05 * 1.461 < 4.5 while 66.7 |Q1+s|^{1+27/164} >= 66.7 * 2^{1.165} > 149
    check("sub-Weyl shifted inequality on 0 <= t <= 3 (trivial margin)", 66.7 * 2**(1 + 27 / 164) > 149 and 3.05 * 1.461 < 4.5, "RHS >= 149.5, LHS <= 4.5")
    # the old shift 1.18 fails for HPY at t = 0 (this is why Q1 = 1.5 is used)
    import mpmath as mpm; mpm.mp.dps = 15
    lhs = abs(mpm.mpc(-0.5, 0) * mpm.zeta(0.5)); rhs = 0.618 * abs(mpm.mpc(1.68, 0))**(7 / 6) * mpm.log(1.68)
    check("Q1 = 1.18 of Bellotti-Wong is NOT admissible for the HPY input at t = 0", lhs > rhs, "LHS %.4f > RHS %.4f" % (lhs, rhs))

def G7(rows):
    print("\nG7  Proposition 1.2, the gains 1 - M(T)/B(T), in ball arithmetic (M: Table 1, B: Bellotti-Wong Table 2)")
    from flint import arb, ctx
    ctx.prec = 200
    R = [(arb(str(w['C1'])), arb(str(w['C2'])), arb(str(w['C3T']))) for w in rows]
    Bq = [(arb(str(a)), arb(str(b)), arb(str(cc))) for a, b, cc in BW_S]
    f = lambda p, L: p[0] * L + p[1] * L.log() + p[2]            # increasing in L (all coefficients > 0)
    def exmin(xs):                                                  # exact minimum of exact balls
        m = xs[0]
        for y in xs[1:]:
            if y < m: m = y
        return m
    Mup = lambda L: exmin([f(p, L).upper() for p in R]);  Mlo = lambda L: exmin([f(p, L).lower() for p in R])
    Bup = lambda L: exmin([f(q, L).upper() for q in Bq]); Blo = lambda L: exmin([f(q, L).lower() for q in Bq])
    # M, B increasing  =>  on [a, b]:  1 - Mup(b)/Blo(a) <= gain <= 1 - Mlo(a)/Bup(b)
    def certify(test):
        stack = [(arb(T0).log(), 200 * arb(10).log())]; n = 0
        while stack:
            a, b = stack.pop(); n += 1
            if test(a, b): continue
            if float((b - a).upper()) < 1e-12: return False, n
            m = (a + b) / 2; stack += [(a, m), (m, b)]
        return True, n
    ok_lo, n_lo = certify(lambda a, b: (1 - Mup(b) / Blo(a)) >= arb('0.00508'))
    ok_hi, n_hi = certify(lambda a, b: (1 - Mlo(a) / Bup(b)) <= arb('0.08965'))
    check("gain >= 0.508%% on [T0, 1e200] (adaptive bisection, %d intervals)" % n_lo, ok_lo)
    check("gain <= 8.965%% on [T0, 1e200] (adaptive bisection, %d intervals)" % n_hi, ok_hi)
    def gain_enclosure(L):
        return 1 - Mup(L) / Blo(L), 1 - Mlo(L) / Bup(L)
    for Ls, lo_ref, hi_ref in (('97.553', '0.0050855', '0.0050866'), ('29.849', '0.0896380', '0.0896390')):
        lo, hi = gain_enclosure(arb(Ls))
        check("gain at L = %s lies in [%s, %s]" % (Ls, lo_ref, hi_ref), (lo >= arb(lo_ref)) and (hi <= arb(hi_ref)),
              "[%s, %s]" % (lo.str(8), hi.str(8)))
    pts = [(arb(T0).log(), '7.4'), (arb(3 * 10**12).log(), '8.7'), (13 * arb(10).log(), '8.9'), (15 * arb(10).log(), '7.0'),
           (20 * arb(10).log(), '3.9'), (30 * arb(10).log(), '1.4'), (40 * arb(10).log(), '0.6'), (200 * arb(10).log(), '3.0')]
    ok = True
    for L, pc in pts:
        lo, hi = gain_enclosure(L); x = arb(pc) / 100
        if not ((lo >= x - arb('0.0005')) and (hi < x + arb('0.0005'))): ok = False
    check("point gains of Proposition 1.2 round to 7.4, 8.7, 8.9, 7.0, 3.9, 1.4, 0.6, 3.0 %", ok)

def G8(rows):
    print("\nG8  Remark 5.1: the intermediate configurations at T0 (parameters from optimize_rows.py)")
    L = math.log(T0); l = math.log(L)
    val = lambda o: min(float(o['C1']) * L + float(o['C2']) * l + float(o['C3T']), float(o['C1']) * L + float(o['C2p']) * l + float(o['C3Tp']))
    bw = min(a * L + b * l + cc for a, b, cc in BW_S)
    check("best bound of Bellotti-Wong Table 2 at T0 rounds to 5.678", round(bw, 3) == 5.678, "%.6f" % bw)
    P = Params(1.182005, 1.682004, 0.182004, n=5, c1=1, c2=1, Q0=1, Qsig=1, dps=20, **PRESET_SUBWEYL_BW)
    v = val(P.constants())
    check("their inputs and shifts, (c,r,eta) = (1.182005, 1.682004, 0.182004): admissible, value rounds to 5.357", P.admissible() and round(v, 3) == 5.357, "%.6f" % v)
    P = Params(1.184687, 1.682269, 0.184686, n=5, k=HPY, c1=1, c2=1, Q0=1, Q1=1.5, Q2=1.5, Q3=6, Q10=2.3, Q11=6, Qsig=1, dps=20)
    v = val(P.constants())
    check("HPY input, c1 = 1, (c,r,eta) = (1.184687, 1.682269, 0.184686): admissible, value rounds to 5.295", P.admissible() and round(v, 3) == 5.295, "%.6f" % v)
    # admissibility of Q3 = Q11 = 6 for the HPY input: |zeta(1/2+it)| <= 1.461 on |t| <= 3 (Hiary) <= 0.618 |6.5+it|^{1/6} log|6.5+it|
    q3 = 0.618 * 6.5**(1 / 6) * math.log(6.5)
    check("shift Q3 = 6 admissible for HPY: 0.618 * 6.5^(1/6) * log 6.5 > 1.461", q3 > 1.461, "%.4f" % q3)
    R1 = (rows[0]['C1'], rows[0]['C2'], rows[0]['C3T']); m = R1[0] * L + R1[1] * l + R1[2]
    check("row 1 of Table 1 at T0 rounds to 5.256", round(m, 3) == 5.256, "%.6f" % m)

if __name__ == "__main__":
    t = time.time(); G1(); G2(); rows = G3(); G4(rows); G5(rows); G6(); G7(rows); G8(rows)
    print("\nPASS=%d FAIL=%d  runtime %.0fs" % (sum(RES), len(RES) - sum(RES), time.time() - t))
