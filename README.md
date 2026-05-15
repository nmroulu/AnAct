# AnAct
Automated KB‑theory workflow for computing solvent activity in 𝑛-component mixtures from MD simulations.

There is no installation pre-requisites. It is enough to download the General_activity_package directory and follow the User_main_script.py script as a guide.
The scripts Activity.py and Gij_pairs_at_TL.py are called within User_main_script.py to compute the activity and the Kirkwood-Buff integrals at the thermodynamic limit, respectively.

MDAnalysis package is used throughout the scripts. Thus, MD trajectories and topologies readable by the MDAnalysis package are expected from the user.

Analyses and examples of the algorithm and its limitations are found in the article soon to be submitted.

An interactive Jupyter-Notebook that prints analytical expressions cen be launched from
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nmroulu/AnAct/HEAD?filepath=Analytical_expressions_notebook%2FAnalytical_Expressions.ipynb)
