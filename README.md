# Copy-Paste Imputation (CPI) for Energy Time Series

This repository contains the Python implementation of the Copy-Paste Imputation (CPI) method presented in the following paper:
>M. Weber, M. Turowski, H. K. Çakmak, R. Mikut, U. Kühnapfel and V. Hagenmeyer, 2021, "Data-Driven Copy-Paste Imputation for Energy Time Series," in IEEE Transactions on Smart Grid, 12, 6, pp. 5409–5419, doi: [10.1109/TSG.2021.3101831](https://doi.org/10.1109/TSG.2021.3101831).

## Installation

To install this project, perform the following steps:
1. Clone the project
2. Open a terminal of the virtual environment where you want to use the project
3. `cd` into the cloned directory
4. `pip install .` or `pip install -e .` to install the project editable.
    * Use `pip install -e .[dev]` to install with development dependencies

## Use

    from cpiets.cpi import CopyPasteImputation
    import pandas as pd

    cpi = CopyPasteImputation()
    data = pd.read_csv('data.csv')
    cpi.fit(data)
    result = cpi.impute()

### Input Data Requirements

**Example data:**

| time                | energy |
| ------------------- | ------:|
| 2012-01-02 00:15:00 |  11.60 |
| 2012-01-02 00:30:00 |  24.87 |
| 2012-01-02 00:45:00 |  37.31 |


The names of the columns are arbitrary.

**Assumptions:**
* There are no missing values (nan) at the start or end of the time series.
* A day starts with the first value after 0:00 (0:15 in the example above) and ends with 0:00.
* The time series starts at the beginning of a day and ends at the end of a day.

**Supported time formats:**
* %Y-%m-%d %H:%M:%S (2020-01-17 13:37:42)
* %d-%b-%Y %H:%M:%S (17-Jan-2020 13:37:42)


## Example

In this repository, we included example data derived from the [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) data set.

To run the CPI method with simple test data, you can run the example

    python example/simple_imputation.py

and play around with the parameters.


## Funding

This project is supported by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy".


## License

This code is licensed under the [LGPL-3.0 License](COPYING).
