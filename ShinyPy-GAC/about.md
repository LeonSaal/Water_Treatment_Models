# Purpose
The purpose of the Drinking Water Treatment and Design Tool (DWTDT) is modelling of multi- or single-solute adsorption of organic chemicals in presence of dissolved organic matter (DOM) in fixed-bed applications. 

# Main Features
The DWTDT facilitates a pore surface diffusion model that offers:

- single-solute adsorption
- multi-solute adsorption (based on ideal adsorbed solution theory)
- empirical correlations for GAC fouling 
- adsorption analysis and tracer model approach to account for DOM in fixed-bed applications
- graphical comparison of different scenarios
- multi-stage modelling

# Technical Details
Underlying numerical tools implemented in Python have been developed by the [US EPA](https://www.epa.gov/), namely:
- [IonExchangeModel](https://github.com/USEPA/Water_Treatment_Models/tree/master/IonExchangeModel) for IX simulation
- [PSDM](https://github.com/USEPA/Water_Treatment_Models/tree/master/PSDM) for GAC simulation

The original PSDM and the IX model by the EPA were slightly modified to fit the purpose of the DWTDT:

The empirical correlation for the liquid phase diffusivity ($D_L$ in m²/s) was changed to account for fictive NOM components with no known molar volume from 
- [Hayduk & Laudie (1974)](https://doi.org/10.1002/aic.690200329):
$D_L = \frac{13.26\cdot 10^{-5}}{\eta^{1.14}\cdot V_b^{0.589}}$ 
- to [Worch (1993)](https://isbnsearch.org/isbn/9783527285662):
$D_L = \frac{3.595\cdot 10^{-9} \cdot T}{\eta\cdot M^{0.53}}$

with dynamic viscosity $\eta$, molar Volume $V_b$, temperature $T$ (in K) and molar mass $M$


## Used packages
- The user interface of the DWTDT was developed using [Shiny for Python](https://shiny.posit.co/py/)
- Unit conversion in the DWTDT is performed using [Pint](https://pint.readthedocs.io/en/stable/)
- Chemical calculations in the DWTDT tool are performed using [molmass](https://github.com/cgohlke/molmass)
- [plotly](https://plotly.com/python-api-reference/) for visualization

# License
The DWTDT can be used under license of the [MIT](https://github.com/USEPA/Water_Treatment_Models/blob/master/LICENSE.md)

# Imprint
Leon Saal (German Environment Agency), Fiona Rückbeil (Berliner Wasserbetriebe), Alexander Sperlich (Berliner Wasserbetriebe)

The DWTDT was developed within [PROMISCES](https://promisces.eu/) (**P**reventing **R**ecalcitrant **O**rganic **M**obile **I**ndustrial chemical**S** for **C**ircular **E**conomy in the **S**oil-sediment-water system). 
The PROMISCES project has received funding from the European Union Horizon 2020 research programme under grant agreement [N°101036449](https://doi.org/10.3030/101036449).



