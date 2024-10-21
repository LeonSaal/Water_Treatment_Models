# Purpose
Modelling of multi- or single-solute adsorption of organic micropollutants in presence of dissolved organic matter

# Main Features
Pores surface diffusion model that offers

- single-solute adsorption

- multi-solute adsorption (ideal adsorbed solution theory)

- empirical correlations for GAC fouling 

- tracer model 

# Technical Details
Underlying numerical tools implemented in Python have been developed by the [US EPA](https://www.epa.gov/) and are used under MIT License namely:
- [IonExchangeModel](https://github.com/USEPA/Water_Treatment_Models/tree/master/IonExchangeModel) for IX simulation
- [PSDM](https://github.com/USEPA/Water_Treatment_Models/tree/master/PSDM) for GAC simulation

They were slightly modified to fit the purpose

This user interface was developed using [Shiny for Python](https://shiny.posit.co/py/)

Unit conversion is performed using [Pint](https://pint.readthedocs.io/en/stable/)

Chemical calculations are performed using [molmass](https://github.com/cgohlke/molmass)