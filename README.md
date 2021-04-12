# A Nonparametric Maximum Likelihood Approach to Mixture of Regressions

This repository contains the implementation for the paper [NPMLE for MR](https://arxiv.org/) in Python. 

## Overview

## Folder
- `scripts/`: scripts that implement the NPMLE procedures (named '\*\_lib.py') and scripts that carry out simulations and real data analysis ('run\_\*.py')
- `data/`: various results in numeric form and storaged in .csv files
- `pics/`: various visualization results
- `real_data/`: data .csv files for real data analysis

## Usage
### Single run
Each script in `scripts/` starting with 'run_' implements one run of a certain numerical experiment. The users can directly run the Python script in their own IDE or alternatively use commands in the terminal. For example, 
```{python}
python run_real_data_gdp.py
```
or
```{python}
python run_simulation.py 
```
When running scripts, certain command-line arugments can be passed to the script in order to switch between different numerical settings. The users can consult the configuration part at the beginning of the scripts.

### Multiple run
There are also scripts that can run multiple simulations with one-line command. For example,
```{python}
python run_multiple_simulations.py
```
There are a few command-line arugments available for this script as well.

## License
All content in this repository is licensed under the MIT license. Comments and suggestions are welcome!
