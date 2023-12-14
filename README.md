![header](images/mpc_simulations_header.png)

Model Predictive Control (MPC) simulations from Melis, Siwanowicz, and
Dickinson 2024

### System Requirements

Tested on ubuntu 22.04 with Python 3.11.4

### Dependencies
The complete list of software dependencies can be found in the "dependencies"
section of the pyproject.toml file. All of the dependencies will be
automatically installed by the Package Installer for Python (pip) when the
software is installed.


### Installation
Requires a working installation of Python. To install cd into the source
Download source and cd into source directory. Then to install using pip run 

```bash
pip install .
```

or to install using [poetry](https://python-poetry.org/) run

```bash
poetry install
```

Software dependencies should be automatically downloaded during the
installation. 

Typical install time is 5-10min.

## Jupyter notebook 

There is no dataset required for this notebook. To Run the notebook

```bash
jupyter notebook MPC_simulations.ipynb
```

### GIFs

---

### Forward Flight
![forward_flight](images/forward_flight.gif)

--- 

### Saccade Right
![saccade_right](images/saccade_right.gif)
