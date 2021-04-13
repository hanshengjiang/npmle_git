# A Nonparametric Maximum Likelihood Approach to Mixture of Regression


## Overview
This repository contains numerical implementation for the paper [NPMLE for MR](https://arxiv.org/). Most scripts are in Python except one R script. These files have been developed and tested in Python version 3.7.4 and R version 3.6.1.

## Folder
- `scripts/`: (1) scripts (named '\*\_lib.py') that implement the NPMLE procedures, including set-ups for simulation and plotting; (2) scripts (named 'run\_\*.py') that carry out simulations and real data analysis
- `data/`: various results stored in .csv files
- `pics/`: various visualization results
- `real_data/`: data .csv files for real data analysis

## Usage
### Single run
Each script starting with 'run_' in `scripts/` is used for one run of a certain numerical experiment. The users can directly run the script in their own IDE or alternatively use commands in the terminal. For example, 
```
python run_real_data_gdp.py
```
or
```
python run_simulation.py 
```
When running scripts, certain command-line arguments can be passed to the script in order to switch among different simulation settings. The users can consult the configuration part at the beginning of the scripts for all arguments available.

### Multiple run
There are also scripts that can run multiple simulations with a one-line command. For example,
```
python run_multiple_simulations.py
```
There are also a few command-line arguments available for this script.

## License
All content in this repository is licensed under the MIT license. Comments and suggestions are welcome!
