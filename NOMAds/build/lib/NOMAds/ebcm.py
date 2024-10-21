import numpy as np
from .iast import calc_c_iast, calc_q_iast, calc_z_iast, iast
from scipy.optimize import fsolve, minimize


def minimize_dq(
    phiqRs, ms: np.array, c0: float, K: float, n: float, qMs: np.array
) -> float:
    """
    calculate MPSD (Marquardts percent standard deviation) for q prediction
    """
    s = 0
    for (phi, qR), m, qM in zip(phiqRs, ms, qMs):
        z = calc_z_iast(c0, m, qR, phi, n, K)
        q = calc_q_iast(z, qR)
        s += np.power((qM - q) / qM, 2)
    mpsd = 100 * np.sqrt(s / (len(qMs) - 2))
    return mpsd


def to_minimize(
    valEBC: tuple,
    valMP: tuple,
    qMPs: np.array,
    ms: np.array,
    Ls: np.array,
    phi0: float = 1e-6,
    qR0: float = 1e-6,
) -> float:
    c0EBC, KEBC, nEBC = valEBC
    c0MP, KMP, nMP = valMP
    c0s = np.concatenate(([c0EBC], [c0MP]))
    ns = np.concatenate(([nEBC], [nMP]))
    Ks = np.concatenate(([KEBC], [KMP]))
    phiqRs = []
    for m, L in zip(ms, Ls):
        res = fsolve(iast, [phi0, qR0], args=(c0s, ns, Ks, m, L))
        phiqRs.append(res)
    return minimize_dq(phiqRs, ms, c0MP, KMP, nMP, qMPs)


def run_ebcm(
    qMPs: np.array,
    cMPs: np.array,
    ms: np.array,
    c0MP: float,
    KMP: float,
    nMP: float,
    L: float = 1,
    fit_KEBC: bool = False,
    fit_nEBC: bool = False,
) -> tuple:
    Ls = L * np.ones_like(ms)
    bounds = (
        (0, c0MP),
        (0, np.inf) if fit_KEBC else (KMP, KMP),
        (0, 1) if fit_nEBC else (nMP, nMP),
    )
    c0EBC, KEBC, nEBC = minimize(
        to_minimize, [0, KMP, nMP], args=((c0MP, KMP, nMP), qMPs, ms, Ls), bounds=bounds
    ).x
    return c0EBC, KEBC, nEBC
