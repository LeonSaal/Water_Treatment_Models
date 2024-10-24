# README for Water Treatment Models
forked from [USEPA/Water Treatment Models](https://github.com/USEPA/Water_Treatment_Models)

Tools in this repository:

1. Adsorption Model for Granular Activated Carbon (PSDM Folder)
2. Ion Exchange Model (IonExchangeModel Folder)
3. Graphical User Interface for Granular Activated Carbon Modeling (ShinyPy-GAC Folder)

Both tools #1 and #2 were programmed by the USEPA in Python. Minor changes were made to #1 to simplify setup as well as usage.
The combined GUI for Ion Exchange Model and PSDM (#3) was newly developed in PYthon with [Shiny for Python](https://shiny.posit.co/py/api/core/). In contrast to the implementations of the USEPA, there is
1. No need for R
2. The possibility to model treatment trains

These tools focus on predicting water treatment unit operation effectiveness, specifically how well treatment technologies (Granular Activated Carbon and Ion Exchange Resins) will work for removing contaminants.

# Status

All code in this repository is being provided in a "draft" state and has not been reviewed or cleared by US EPA. This status will be updated as models are reviewed.

# Additional Information

See also tools found at https://github.com/USEPA/Environmental-Technologies-Design-Option-Tool

This repository is released under the [MIT License](LICENSE.md).

# EPA Disclaimer

The United States Environmental Protection Agency (EPA) GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. EPA has relinquished control of the information and no longer has responsibility to protect the integrity , confidentiality, or availability of the information. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by EPA. The EPA seal and logo shall not be used in any manner to imply endorsement of any commercial product or activity by EPA or the United States Government.

By submitting a pull request, you make an agreement with EPA that you will not submit a claim of compensation for services rendered to EPA or any other federal agency. Further, you agree not to charge the time you spend developing software code related to this project to any federal grant or cooperative agreement.