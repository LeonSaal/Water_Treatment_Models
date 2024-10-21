import array
import numpy as np
from scipy.optimize import minimize, curve_fit, fsolve
from .iast import calc_q_iast, calc_z_iast, iast, calc_c_iast
from .models import freundlich


def run_trm(
    cMPs: np.array,
    ms: np.array,
    c0MP: float,
    c0sDOC: np.array,
    KsDOC: np.array,
    nsDOC: np.array,
    L: float = 1,
) -> tuple[float, float]:
    """_summary_

    Args:
        cMPs (np.array): carbon concentrations of micropollutant ([massC]/[volume])
        ms (np.array): adsorbent dosages
        c0MP (float): initial carbon concentration of micropollutant ([massC]/[volume])
        c0sDOC (np.array): initial concentrations of fictive NOM components as calculated by ADSA
        KsDOC (np.array): Freundlich K's for fictive NOM components
        nsDOC (np.array): Freundlich n's for fictive NOM components
        L (float, optional): volume. Defaults to 1.

    Returns:
        tuple[float, float]: (fitted K for micropollutant (([massC]/[mass])/(([massC]/[volume])**n)), fitted n for micropollutant)
    """
    # fit pseudo single solute isotherm for starting values
    qMPs = (c0MP - cMPs) / ms
    (K0MP, n0MP), _ = curve_fit(freundlich, cMPs, qMPs, bounds=([0, 0], [np.inf, 1]))
    c0s = np.concatenate((c0MP, c0sDOC))
    Ls = L * np.ones_like(c0s)

    # get modified K and n for micropollutant (MP)
    KMP, nMP = minimize(
        to_minimize,
        np.array([K0MP, n0MP]),
        args=(c0s, nsDOC, KsDOC, ms, Ls, qMPs),
        bounds=((0, np.inf), (0, 1)),
    ).x

    return KMP, nMP


def minimize_dq(
    phiqRs: list, ms: np.array, c0: float, K: float, n: float, qMs: np.array
) -> float:
    """calculate MPSD (Marquardts percent standard deviation) for q prediction
        as described by Kumar et al. 10.1016/j.jhazmat.2007.09.020

    Args:
        phiqRs (list): Phi and qR from IAST
        ms (np.array): adsorbent dosages
        c0 (float): initial concentration
        K (float): Freundlich K
        n (float): Freundlich n
        qMs (np.array): measured loadings

    Returns:
        float: MPSD
    """
    s = 0
    for (phi, qR), m, qM in zip(phiqRs, ms, qMs):
        z = calc_z_iast(c0, m, qR, phi, n, K)
        q = calc_q_iast(qR, z)
        s += np.power((qM - q) / qM, 2)
    mpsd = 100 * np.sqrt(s / (len(qMs) - 2))
    return mpsd


def term_s3_6(
    phiqRs: list, ms: np.array, c0: float, K: float, n: float, qMs: np.array
) -> float:
    """term to minimize after "johannsenMathematischeMethodeZur1994", 10.1002/aheh.19940220504 (6)

    Args:
        qRs (np.array): calculated loadings
        qMs (np.array): measured loadings
        cTs (np.array): total concentrations
        Ls (np.array): volumes
        ms (np.array): adsorbent dosages

    Returns:
        float: S3 from 10.1002/aheh.19940220504
    """
    s3 = 0
    for (phi, qR), m, qM in zip(phiqRs, ms, qMs):
        z = calc_z_iast(c0, m, qR, phi, n, K)
        q = calc_q_iast(qR, z)
        c = calc_c_iast(c0, q, m)
        cM = calc_c_iast(c0, qM, m)
        dq = np.abs((qM - q) / qM)
        dc = np.abs((cM - c) / cM)
        s3 += np.power(dq + dc, 2)
    return s3


def to_minimize(
    KMPnMP: np.array,
    c0s: np.array,
    nsDOC: np.array,
    KsDOC: np.array,
    ms: np.array,
    Ls: np.array,
    qMPs: np.array,
    phi0: float = 1e-6,
    qR0: float = 1e-6,
) -> float:
    """_summary_

    Args:
        KMPnMP (np.array): Freundlich K and n for micropollutant
        c0s (np.array): initial concentrations of all components in system 1st micropollutant, then fictive NOM
        nsDOC (np.array): Freundlich n's for fictive NOM components
        KsDOC (np.array): Freundlich K's for fictive NOM components
        ms (np.array): adsorbent dosages
        Ls (np.array): volumes
        qMPs (np.array): measured loadings of micropollutant
        phi0 (float, optional): initial Phi for IAST fit. Defaults to 1e-6.
        qR0 (float, optional): initial qR for IAST fit Defaults to 1e-6.

    Returns:
        float: value to minimize to find best values for K and n for micropollutant
    """
    KMP, nMP = KMPnMP
    # merge values to be optimized to other
    ns = np.concatenate(([nMP], nsDOC))
    Ks = np.concatenate(([KMP], KsDOC))
    phiqRs = []
    # perform iast calculation to get phi and qR
    for m, L in zip(ms, Ls):
        res = fsolve(iast, [phi0, qR0], args=(c0s, ns, Ks, m, L))
        phiqRs.append(res)
    return term_s3_6(phiqRs, ms, c0s[0], KMP, nMP, qMPs)
