import pandas as pd
import numpy as np
from scipy.optimize import fsolve


# iast to get values for phi and qR (qT)
def iast(
    phiqR: np.array, c0s: np.array, ns: np.array, Ks: np.array, m: float, L: float
) -> list[float, float]:
    """

    Args:
        phiqR (np.array): Phi and qR from IAST
        c0s (np.array): initial molar concentrations of all components
        ns (np.array): Freundlich n's of all components in system
        Ks (np.array): Freundlich K's of all components in system
        m (float): adsorbent dosage
        L (float): volume

    Returns:
        list[float, float]: values of two equations that are supposed to be equal to zero
    """
    phi, qR = np.abs(phiqR)
    g1 = 1
    g2 = phi / qR
    z1 = 0
    z2 = 0
    for c0, n, K in zip(c0s, ns, Ks):
        z = c0 / (np.power(phi * n / K, 1 / n) + qR * m / L)
        z1 += z
        z2 += z * 1 / n
    g1 -= z1
    g2 -= z2
    return [g1, g2]


# helper functions for predict_iast_one
def calc_z_iast(
    c0: float, m: float, qR: float, phi: float, n: float, K: float
) -> float:
    """calculate z from IAST

    Args:
        c0 (float): initial molar concentration of component of interest
        m (float): adsorbent dosage
        qR (float): qR from IAST fit
        phi (float): Phi from IAST fit
        n (float): Freundlich n of component
        K (float): Freundlich K of component

    Returns:
        float: z
    """
    return c0 / (m * qR + np.power(phi * n / K, 1 / n))


def calc_q_iast(qR: float, z: float) -> float:
    """Calculate q from IAST

    Args:
        qR (float): qR from IAST
        z (float): z of component

    Returns:
        float: loading q
    """
    return qR * z


def calc_c_iast(c0: float, q: float, m: float) -> float:
    """Calculate c from IAST fit via mass balance

    Args:
        c0 (float): initial concentration of component
        q (float): loading
        m (float): adsorbent dosage

    Returns:
        float: residual liquid phase concentration
    """
    return c0 - q * m


# calculate equilibrium with known values for c0, n, K, ...
def predict_iast_once(
    c0s: np.array,
    ns: np.array,
    Ks: np.array,
    m: float,
    L: float,
    phi0: float = 1e-6,
    qR0: float = 1e-6,
):
    res = fsolve(iast, [phi0, qR0], args=(c0s, ns, Ks, m, L))
    phi, qR = res

    cis = []
    qis = []
    zis = []
    for c0, n, K in zip(c0s, ns, Ks):
        z = calc_z_iast(c0, m, qR, phi, n, K)
        q = calc_q_iast(qR, z)
        c = calc_c_iast(c0, q, m)
        zis.append(z)
        qis.append(q)
        cis.append(c)

    return phi, qR, cis, qis, zis


def predict_iast_multi(
    c0R: np.array, ns: np.array, Ks: np.array, ms: np.array, Ls: np.array
) -> pd.DataFrame:
    cs = []
    qs = []
    zs = []
    phis = []
    qRs = []
    phi0 = (Ks / ns * c0R**ns).mean()
    qR0 = phi0 / 2
    for m, L in zip(ms, Ls):
        phi, qR, cis, qis, zis = predict_iast_once(c0R, ns, Ks, m, L, phi0=phi0, qR0=qR0)
        cs.append(cis)
        qs.append(qis)
        zs.append(zis)
        phis.append(phi)
        qRs.append(qR)

    df0 = pd.DataFrame([c0R], columns=[f"c{i}" for i, _ in enumerate(c0R)])
    df0["cges"] = c0R.sum()

    df = pd.DataFrame(columns=["m", "L", "qR", "phi"])
    df.m = ms
    df.L = Ls
    df.qR = qRs
    df.phi = phis

    df = pd.concat(
        [df]
        + [
            pd.DataFrame(data, columns=[f"{name}{i}" for i in range(len(data[0]))])
            for data, name in zip([qs, zs, cs], ["q", "z", "c"])
        ],
        axis=1,
    )
    df["cges"] = np.array(cs).sum(axis=1)

    return pd.concat([df0, df], axis=0, ignore_index=True)
