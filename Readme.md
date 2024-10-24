# Water Treatment Models
forked from [USEPA/Water Treatment Models](https://github.com/USEPA/Water_Treatment_Models)

Tools in this repository:

1. Adsorption Model for Granular Activated Carbon (PSDM Folder)
2. Ion Exchange Model (IonExchangeModel Folder)
3. Graphical User Interface for Granular Activated Carbon Modeling (ShinyPy-GAC Folder)

Both tools #1 and #2 were programmed by the USEPA in Python. Minor changes were made to #1 to simplify setup as well as usage.
The combined GUI for Ion Exchange Model and PSDM (#3) was newly developed in Python with [Shiny for Python](https://shiny.posit.co/py/api/core/). In contrast to the implementations of the USEPA, there is
1. No need for R
2. The possibility to model treatment trains
3. compare scenarios

These tools focus on predicting water treatment unit operation effectiveness, specifically how well treatment technologies (Granular Activated Carbon and Ion Exchange Resins) will work for removing contaminants.

# Status

All code in this repository is being provided in a "draft" state and has not been reviewed or cleared by US EPA. This status will be updated as models are reviewed.

# Additional Information

See also tools found at https://github.com/USEPA/Environmental-Technologies-Design-Option-Tool

This repository is released under the [MIT License](LICENSE.md).

# Setup
1. Download or clone repository 

2. with e.g. conda make new environent and navigate to this folder
In Anaconda promt:
    ```
    conda create -n <YOUR ENVIRONMENT> python
    conda activate <YOUR ENVIRONMENT>
    cd <HERE>
    ```

3. Install requirements
    ```
    pip install -r requirements.txt
    ```

# Usage
Run app and use in browser
```
shiny run -b ShinyPy-GAC/app.py
```
