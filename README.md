# NPMLE: Nonparametric Maximum Likelihood Estimation of Mixture of Regression

## Overview
This repository contains reproduction code and data for numerical implementation in the paper [A Nonparametric Maximum Likelihood Approach to Mixture of Regression](https://arxiv.org/abs/2108.09816). Most scripts are in Python except one R script. These files have been developed and tested in Python version 3.7.4 and R version 3.6.1.

## Folder
- `scripts/`: (1) scripts (named '\*\_lib.py') that implement the NPMLE procedures, including set-ups for simulation and plotting; (2) scripts (named 'run\_\*.py') that carry out simulations and real data analysis
- `data/`: various results stored in .csv files
- `pics/`: various visualization results
- `real_data/`: data .csv files for real data analysis

## Usage
### Single run
Each script starting with 'run_' in `scripts/` is used for one run of a certain numerical experiment. The users can directly run the script in their own IDE but running the script from the terminal is recommended. For example, 
```
python run_real_data_gdp.py
```
or
```
python run_simulation.py 
```
When running scripts in terminal, certain command-line arguments can be passed to the script in order to switch among different simulation settings. For example, for script `run_simulation.py` one can specify a set of command line arguments as below, which represent that scale parameter = `0.5`, number of data points = `500`, component coefficients setup = type `1`, whether to run cross-validation = `yes`, and the granularity of cross-validation = `0.01` respectively.
```
python run_simulation.py 0.5 500 1 yes 0.01
```

The users can consult the configuration part at the beginning of the scripts to find all arguments that are available.

### Multiple run
There are also scripts in `scripts/` that can run multiple simulations with a one-line command. For example, one can use the script `run_multiple_simulations.py` to run all simulation examples for discrete cases by adding two command-line arguments as follows.
```
python run_multiple_simulations.py run_simulation.py discrete_cv
```
There are also a few other command-line arguments available for this script, which are specified in documentation.

## License
All content in this repository is licensed under the MIT license. Comments and suggestions are welcome!
