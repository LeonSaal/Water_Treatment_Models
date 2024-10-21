from scipy.optimize import fsolve, minimize
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from .iast import iast, predict_iast_multi


def term_s3(
    qRs: np.array, qMs: np.array, cTs: np.array, Ls: np.array, ms: np.array
) -> float:
    """term to minimize after "johannsenMathematischeMethodeZur1994", 10.1002/aheh.19940220504 (8)

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
    for qR, qM, L, m, cT in zip(qRs, qMs, Ls, ms, cTs):
        s3 += np.power((qR - qM) * (1 / qM + 1 / (cT * L / m - qM)), 2)

    return s3


def to_minimize(
    c0s: np.array,
    ns: np.array,
    Ks: np.array,
    qMs: np.array,
    ms: np.array,
    Ls: np.array,
    cTs: np.array,
    phi0: float = 1e-6,
    qR0: float = 1e-6,
) -> float:
    """_summary_

    Args:
        c0s (np.array): initial concentrations of fictive components
        ns (np.array): Freundlich n's
        Ks (np.array): Freundlich K's
        qMs (np.array): measured loadings
        ms (np.array): adsorbent dosages
        Ls (np.array): volumnes
        cTs (np.array): total concentrations 
        phi0 (float, optional): initial spreading pressure Phi. Defaults to 1e-6.
        qR0 (float, optional): initial total loading qR. Defaults to 1e-6.

    Returns:
        float: value calculated by term_s3()
    """
    qRs = []
    for m, L in zip(ms, Ls):
        res = fsolve(iast, [phi0, qR0], args=(c0s, ns, Ks, m, L))
        qRs += [res[1]]

    return term_s3(qRs, qMs, cTs, Ls, ms)


def run_adsa(
    cT0: np.array,
    cTs: np.array,
    ms: np.array,
    qMs: np.array,
    name: str,
    Ks: list = np.array([0, 20, 40, 60]),
    n: float = 0.25,
    L: float = 1,
    plot: bool = True,
    path: str = "",
) -> pd.DataFrame:
    """_summary_

    Args:
        cT0 (np.array): initial concentration
        cTs (np.array): residual solute concentration
        ms (np.array): adsorbent dosages
        qMs (np.array): measured total loadings
        name (str): name to use for output
        Ks (list, optional): Freundlich K's to use for fictive components. Defaults to [0, 20, 40, 60].
        n (float, optional): Freundlich n to use for fictive components. Defaults to 0.25.
        L (float, optional): volume. Defaults to 1.
        plot (bool, optional): show results. Defaults to True.
        path (str, optional): save results to path. Defaults to "" for no output.

    Returns:
        pd.DataFrame: details of IAST calculation and fitting results for c_i(m=0)
    """
    # model input
    Ls = L * np.ones_like(ms)
    ## adsorption parameters
    ns = n * np.ones_like(Ks)
    ## starting values for fictive DOC component
    c0s = cT0 / len(Ks) * np.ones_like(Ks)

    # boundary conditions
    ## 0 < c0 < cT
    bounds = ((0, cT0) for _ in c0s)
    ## sum(c0) <= cT
    constraints = {"type": "eq", "fun": lambda x: sum(x) - cT0}

    # starting values for phi and qR
    phi0 = (Ks/ns * cT0**ns).mean()
    qR0 = qMs.mean()

    # find c0s
    c0R = minimize(
        lambda x: to_minimize(x, ns, Ks, qMs, ms, Ls, cTs, qR0=qR0, phi0=phi0),
        c0s,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    ).x

    # calculate equilibrium from fitted values
    df = predict_iast_multi(c0R, ns, Ks, ms, Ls)
    sample = pd.DataFrame(
        {
            "dosage": np.concatenate(([0], ms)),
            "doc": np.concatenate((cT0, cTs)),
            "qM": np.concatenate(([np.nan], qMs)),
        }
    )
    fulldata = pd.concat([sample.reset_index(drop=True), df], axis=1)
    Ksc = [str(K) for K in Ks]
    if plot:
        fig, ax = plt.subplots()

        # draw data
        ax.scatter(fulldata.cges, fulldata.qR, label="IAST")
        ax.scatter(fulldata.doc, fulldata.qM, label="measured")

        # set labels
        ax.set_title(
            f"{name}\nn={n}, K:c0 = "
            + "; ".join([f"{K}:{c0:.2f}" for K, c0 in zip(Ks, c0R)])
        )
        ax.set_xlabel("c /mg/L")
        ax.set_ylabel("q /mg/g")
        ax.legend()

        # save figure
        if path:
            outfile = os.path.join(path, f"adsa_{name}_n_{n}_K_{'_'.join(Ksc)}.png")
            fig.savefig(outfile)

    if path:
        outfile = os.path.join(path, f"adsa_{name}.xlsx")
        kwargs = {"if_sheet_exists": "replace"} if os.path.exists(outfile) else {}
        with pd.ExcelWriter(
            outfile,
            mode=("a" if os.path.exists(outfile) else "w"),
            engine="openpyxl",
            **kwargs,
        ) as writer:
            fulldata.to_excel(writer, sheet_name=f"{name}_n_{n}_K_{'_'.join(Ksc)}")

    return fulldata
