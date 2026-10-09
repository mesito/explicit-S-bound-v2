"""sbound_arb.py -- CERTIFIED VERSION (python-flint / Arb ball arithmetic, rigorous quadrature) -- explicit constants for |S(T)| <= C1 log T + C2 loglog T + C3~ (T >= T0)
in the framework of Bellotti and Wong, "Improved estimates for the argument and
zero-counting function of the Riemann zeta-function", Math. Comp. (2025),
Theorems 1.2 and 1.4, formulas (3.11)-(3.15) and (4.1)-(4.2).

Arb / python-flint implementation: ball arithmetic at `prec` bits with rigorous quadrature (acb_calc_integrate).
Reproduces the five rows of their Table 2 (see verify_S_bound.py).

Inputs (each with its own range; what the method consumes are the shifted forms, valid for all real t):
  |zeta(1/2+it)|      <= k1 t^k2 (log t)^k3   (t >= 3; HPY: 0.618, 1/6, 1 ; Patel-Yang: 66.7, 27/164, 0;
                                               shifted form with Q1 = Q2 = 1.5, certified for |t| <= 3 in verify_S_bound.py G6)
  |zeta(1+it)|        <= c1 (log t)^c2        (t >= t0 = Q0; here c2 = 1 and c1 = c1(Q0) from Hoo-Teo, see c1_of_Q0;
                                               shifted form |it zeta(1+it)| <= c1 |Q0+1+it| log|Q0+1+it| for all t)
  |zeta(sigma_k+it)|  <= 1.546 t^{1/(2^k-2)} log t  (t >= 3; Yang), sigma_k = 1 - k/(2^k-2), k = 4..n+4
"""
from flint import arb, acb, ctx
import math

def bmin(a, b):
    """rigorous min of two balls"""
    return (a + b - abs(a - b)) / 2
def bmax0(a):
    return (a + abs(a)) / 2
def A(x):
    """arb from int / float (via decimal string) / fraction string 'p/q' / arb"""
    if isinstance(x, arb): return x
    if isinstance(x, str) and '/' in x:
        p, q = x.split('/'); return arb(p) / arb(q)
    return arb(str(x)) if isinstance(x, float) else arb(x)
def c1_of_Q0(Q0):
    """c1 such that 1/2 log t + 0.6633 <= c1 log|Q0+1+it| for all t >= 3 (Hoo-Teo 2026, valid for t >= e)."""
    Q0 = A(Q0)
    return arb('0.5') + arb('0.6633') / Q0.log()

class Params:
    def __init__(self, c, r, eta, n=5, k=('0.618', '1/6', 1), c1=1, c2=1,
                 Q0=1, Q1=1.5, Q2=1.5, Q3=1e9, Qsig=1, Q10=1e9, Q11=1e9,
                 T0=30610046000, J1=64, J2=39, dps=30, prec=200):
        ctx.prec = prec
        f = A
        self.c, self.r, self.eta, self.n = f(c), f(r), f(eta), int(n)
        self.k1, self.k2, self.k3 = f(k[0]), f(k[1]), f(k[2])
        self.c1, self.c2 = f(c1), f(c2)
        self.Q0, self.Q1, self.Q2, self.Q3, self.Qsig, self.Q10 = f(Q0), f(Q1), f(Q2), f(Q3), f(Qsig), f(Q10)
        self.Q11 = f(Q11) if Q11 is not None else max(self.Q3, self.Q10)
        self.T0, self.J1, self.J2 = f(T0), int(J1), int(J2)
        self.sigma1 = self.c + (self.c - f('0.5'))**2 / self.r
        self.delta = 2 * self.c - self.sigma1 - f('0.5')
        self.sig = [1 - arb(4 + h) / (2**(4 + h) - 2) for h in range(self.n + 1)]   # sigma_{4+h}
        self.L = (self.r / (self.c - f('0.5'))).log()

    # -- geometry
    def theta(self, y):
        y = A(y)
        if self.c + self.r <= y: return arb(0)
        if (self.c - self.r < y) and (y < self.c + self.r): return ((y - self.c) / self.r).acos()
        return arb.pi()
    def sigma(self, th): return self.c + self.r * th.cos()
    def q(self, f, a, b):
        """rigorous integral of an entire integrand over [a,b] (acb_calc_integrate); returns an arb enclosure"""
        if a == b: return arb(0)
        return acb.integral(lambda x, analytic: f(x), acb(a), acb(b)).real
    def Lstar(self, th, j):
        return (j + self.c + self.r * th.cos())**2 / self.T0 + (self.r * th.sin())**2 / self.T0 + 2 * self.r * th.sin()

    def admissible(self):
        c, r, eta, s1, d = self.c, self.r, self.eta, self.sigma1, self.delta
        half = arb('0.5'); quarter = arb('0.25')
        return bool((c - r > -half) and (c - r < 1 - c) and (1 - c < -eta) and (eta > 0) and (eta <= half)
                and (d >= quarter) and (d < half) and (1 + eta < c) and (s1 > c) and (s1 < c + r)
                and (self.theta(1 + eta) <= arb('2.1')))

    def constants(self):
        f = A; n, s, th, sg, q = self.n, self.sig, self.theta, self.sigma, self.q
        k1, k2, k3, c1, c2, eta = self.k1, self.k2, self.k3, self.c1, self.c2, self.eta
        N4 = 2**(n + 4)
        # ---- C1 (3.11)
        C1b = q(lambda w: (N4 - 1) * (1 - sg(w)) + (N4 - 2) * (sg(w) - s[n]), th(1), th(s[n])) / ((N4 - 2) * (1 - s[n]))
        for h in range(n):
            a, b = 2**(h + 4) - 2, 2**(h + 5) - 2
            C1b += q(lambda w, h=h, a=a, b=b: b * (a + 1) * (s[1 + h] - sg(w)) + a * (b + 1) * (sg(w) - s[h]),
                     th(s[1 + h]), th(s[h])) / (a * b * (s[1 + h] - s[h]))
        C1b += q(lambda w: 14 * (k2 + 1) * (s[0] - sg(w)) + 15 * (sg(w) - f('0.5')), th(s[0]), th(f('0.5'))) / (14 * (s[0] - f('0.5')))
        C1b += f('0.5') * q(lambda w: 1 - 2 * sg(w) + 4 * k2 * sg(w), th(f('0.5')), th(0))
        C1b += f('0.5') * q(lambda w: 1 - 2 * sg(w), th(-eta), arb.pi())
        C1b += q(lambda w: -sg(w) * (1 + 2 * eta) / (2 * eta) + (sg(w) + eta) / (2 * eta), th(0), th(-eta))
        C1b += th(1) - th(f('0.5'))
        C1 = C1b / (2 * arb.pi() * self.L)
        # ---- C2 (3.12), C2' (3.13)
        C2b = q(lambda w: (1 - sg(w)) + c2 * (sg(w) - s[n]), th(1), th(s[n])) / (1 - s[n])
        C2b += th(s[0]) - th(s[n])
        C2b += q(lambda w: k3 * (s[0] - sg(w)) + (sg(w) - f('0.5')), th(s[0]), th(f('0.5'))) / (s[0] - f('0.5'))
        C2b += (c2 / eta) * q(lambda w: 1 + eta - sg(w), th(1 + eta), th(1))
        C2b += q(lambda w: c2 * (1 - 2 * sg(w)) + 2 * k3 * sg(w), th(f('0.5')), th(0))
        C2b += q(lambda w: c2 * (sg(w) + eta) / eta, th(0), th(-eta))
        C2 = C2b / (2 * arb.pi() * self.L)
        C2p = C2 + f('2.00204') / (2 * self.L)
        # ---- D3 (with the 1.00212 normalisation)
        lc1 = (c1 * f('1.00212')**c2).log(); lY = (f('1.546') * f('1.00212')).log(); lk1 = (k1 * f('1.00212')**k3).log()
        D3 = (lc1 / eta) * q(lambda w: 1 + eta - sg(w), th(1 + eta), th(1)) + ((1 + eta).zeta().log() / eta) * q(lambda w: sg(w) - 1, th(1 + eta), th(1))
        D3 += (lY / (1 - s[n])) * q(lambda w: 1 - sg(w), th(1), th(s[n])) + (lc1 / (1 - s[n])) * q(lambda w: sg(w) - s[n], th(1), th(s[n]))
        D3 += lY * (th(s[0]) - th(s[n]))
        D3 += (lk1 / (s[0] - f('0.5'))) * q(lambda w: s[0] - sg(w), th(s[0]), th(f('0.5'))) + (lY / (s[0] - f('0.5'))) * q(lambda w: sg(w) - f('0.5'), th(s[0]), th(f('0.5')))
        D3 += (c1 * f('1.00212')**c2 / (2 * arb.pi()).sqrt()).log() * q(lambda w: 1 - 2 * sg(w), th(f('0.5')), th(0)) + 2 * lk1 * q(lambda w: sg(w), th(f('0.5')), th(0))
        D3 += q(lambda w: -(sg(w) / eta) * ((1 + eta) / (c1 * (2 * arb.pi())**eta)).log() + (c1 * f('1.00212')**c2 / (2 * arb.pi()).sqrt()).log(), th(0), th(-eta))
        D3 += -((2 * arb.pi()).log() / 2) * q(lambda w: 1 - 2 * sg(w), th(-eta), arb.pi())
        lz = ((1 + eta).zeta().log() + self.c.zeta().log()) / 2
        D3 += lz * (th(1 + eta) - arb.pi() / 2) + (arb.pi() / (4 * self.J1)) * self.c.zeta().log()
        D3 += lz * (th(1 - self.c) - th(-eta)) + ((arb.pi() - th(1 - self.c)) / (2 * self.J2)) * self.c.zeta().log()
        # ---- kappa1, kappa2 (Lemma 4.3 of HSW22)
        kap1 = (arb.pi() / (4 * self.J1)) * ((self.c + self.r).zeta().log() + 2 * sum(((self.c + self.r * (arb.pi() * j / (2 * self.J1)).cos()).zeta().log() for j in range(1, self.J1)), arb(0)))
        t1c = th(1 - self.c)
        kap2 = ((arb.pi() - t1c) / (2 * self.J2)) * ((1 - self.c + self.r).zeta().log() + 2 * sum(((1 - self.c - self.r * (arb.pi() * j / self.J2 + (1 - arb(j) / self.J2) * t1c).cos()).zeta().log() for j in range(1, self.J2)), arb(0)))
        # ---- kappa3 (finite-T0 terms)
        Ls = self.Lstar; Q0, Q2, Q10, Q11 = self.Q0, self.Q2, self.Q10, self.Q11; Q0n = max(Q0, self.Qsig)
        M1 = q(lambda w: Ls(w, -1), 0, th(1 + eta)) + q(lambda w: Ls(w, Q0), th(1 + eta), th(1))
        M1 += q(lambda w: ((N4 - 1) * (1 - sg(w)) + (N4 - 2) * (sg(w) - s[n])) * Ls(w, Q0n), th(1), th(s[n])) / ((N4 - 2) * (1 - s[n]))
        for h in range(n):
            a, b = 2**(h + 4) - 2, 2**(h + 5) - 2
            M1 += q(lambda w, h=h, a=a, b=b: (b * (a + 1) * (s[1 + h] - sg(w)) + a * (b + 1) * (sg(w) - s[h])) * Ls(w, self.Qsig), th(s[1 + h]), th(s[h])) / (a * b * (s[1 + h] - s[h]))
        M1 += q(lambda w: (14 * (k2 + 1) * (s[0] - sg(w)) + 15 * (sg(w) - f('0.5'))) * Ls(w, Q2), th(s[0]), th(f('0.5'))) / (14 * (s[0] - f('0.5')))
        M1 += q(lambda w: Ls(w, -1), th(f('0.5')), th(0)) + f('0.5') * q(lambda w: (1 - 2 * sg(w) + 4 * k2 * sg(w)) * Ls(w, Q11), th(f('0.5')), th(0))
        M1 += q(lambda w: Ls(w, -1), th(0), th(-eta)) + q(lambda w: (-sg(w) + f('0.5')) * Ls(w, Q10), th(0), th(-eta))
        M1 += q(lambda w: Ls(w, -1), th(-eta), arb.pi()) + q(lambda w: f('0.5') * (1 - 2 * sg(w)) * Ls(w, 1), th(-eta), arb.pi())
        M2 = q(lambda w: (c2 / eta) * (1 + eta - sg(w)) * Ls(w, Q0), th(1 + eta), th(1))
        M2 += q(lambda w: ((1 - sg(w)) + c2 * (sg(w) - s[n])) * Ls(w, Q0n), th(1), th(s[n])) / (1 - s[n])
        for h in range(n):
            M2 += q(lambda w: Ls(w, self.Qsig), th(s[1 + h]), th(s[h]))
        M2 += q(lambda w: (k3 * (s[0] - sg(w)) + (sg(w) - f('0.5'))) * Ls(w, Q2), th(s[0]), th(f('0.5'))) / (s[0] - f('0.5'))
        M2 += q(lambda w: (c2 * (1 - 2 * sg(w)) + 2 * k3 * sg(w)) * Ls(w, Q11), th(f('0.5')), th(0))
        M2 += q(lambda w: c2 * (sg(w) + eta) / eta * Ls(w, Q10), th(0), th(-eta))
        kap3 = bmax0(M1) / (2 * self.T0) + bmax0(M2) / (2 * self.T0 * self.T0.log())
        # ---- C3 (3.14), C3' (3.15), tilde versions (4.1), (4.2)
        base = arb(7) / 8 + arb(1) / 4 + 1 / (50 * self.T0) + self.sigma1.zeta().log() / arb.pi() + f('0.5') * ((640 * self.delta - 112) / (1536 * (3 * self.T0 - 1)) + arb(2)**-10)
        tail = (D3 + kap1 + kap2 + kap3) / (2 * arb.pi() * self.L)
        C3 = base + (self.c.zeta() / (2 * self.c).zeta()).log() / (2 * self.L) + tail
        C3p = base + tail
        corr = -f(7) / 8 - 1 / (50 * self.T0) + (((self.sigma1 - 1) / self.T0).atan() + (1 / (2 * self.T0)).atan()) / arb.pi()
        return dict(C1=C1, C2=C2, C2p=C2p, C3=C3, C3p=C3p, C3T=C3 + corr, C3Tp=C3p + corr, kap3=kap3)

SUBWEYL = ('66.7', '27/164', 0)
HPY = ('0.618', '1/6', 1)

# Named presets of (input, shifts); see sbound.py.  Defaults of Params are those of PRESET_HPY.
PRESET_HPY = dict(k=HPY, Q1='1.5', Q2='1.5', Q3=10**9, Q10=10**9, Q11=10**9)
PRESET_SUBWEYL_BW = dict(k=SUBWEYL, Q1='1.18', Q2='1.18', Q3='3.9', Q10='2.3', Q11='3.9')
