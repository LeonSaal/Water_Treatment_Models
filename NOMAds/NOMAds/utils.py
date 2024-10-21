from pint import UnitRegistry
from molmass import Formula
import re


def convert_K(K: float, n: float, from_unit: str, to_unit: str, formula: str = "C") -> float:
    """Convert Freundlich K between units. 
        Molar concentrations and carbon concentrations ae also possible 
        if sum formula of analyte is supplied

    Args:
        K (float): Freundlich K to convert
        n (float): Freundlich n
        from_unit (str): unit to convert from. Exponent n must be specified using brackets, e.g. '(mg(/L))/((mg/L)**{n})'
        to_unit (str): unit to convert from. Exponent n must be specified using brackets, e.g. '(mg(/L))/((mg/L)**{n})'
        formula (str, optional): formula of analyte. Necessary if K is converted from or to molar concentrations or carbon concentrations. Defaults to "C".

    Returns:
        float: K converted to unit 'to_unit'
    """
    ureg = UnitRegistry()
    # get mass and carbon content
    formula = Formula(formula)
    MW = formula.mass
    comp = formula.composition()
    if "C" in comp:
        wC = comp["C"].mass / MW
        ureg.define(f"gC = g / {wC}")

    ureg.define(f"mol = g * {MW}")

    # split fraction for seperate conversions due to float precision error
    pattern = r"(?<=\))\s*/\s*(?=\()"
    enum, denom = [
        ureg(ufrom.format(n=n)).to(uto.format(n=n)).magnitude
        for ufrom, uto in zip(re.split(pattern, from_unit), re.split(pattern, to_unit))
    ]
    return K * enum / denom


def mass_to_massC_or_mol(from_unit: str, to_unit: str, formula: str = "C")-> float:
    """Convert mass (-concentrations) to molar or carbon and vice versa.

    Args:
        from_unit (str): unit to convert from
        to_unit (str): unit to convert to
        formula (str, optional): formula of analyte. Necessary if K is converted from or to molar or carbon values. Defaults to "C".

    Returns:
        float: conversion factor so that: value[unit_from] * factor = value[unit_to] 
    """
    ureg = UnitRegistry()
    # get mass and carbon content
    formula = Formula(formula)
    MW = formula.mass
    ureg.define(f"mol = (g * {MW})")

    comp = formula.composition()
    if "C" in comp:
        wC = comp["C"].mass / MW
        ureg.define(f"gC =(g / {wC})")

    return ureg(from_unit).to(to_unit).magnitude
