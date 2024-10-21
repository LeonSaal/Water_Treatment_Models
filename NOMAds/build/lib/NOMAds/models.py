import numpy as np

def freundlich(c: np.array, K: float, n: float) -> np.array:
    """_summary_

    Args:
        c (np.array): concentrations
        K (float): Freundlich K
        n (float): Freundlich n

    Returns:
        np.array: q(c)
    """
    return K * np.power(c, n)