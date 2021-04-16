# Copy-Paste Imputation for Energy Time Series

## Installation

To install this project, perform the following steps:
1. Clone the project
2. Open a terminal of the virtual environment where you want to use the project
3. `cd` into the cloned directory
4. `pip install .` or `pip install -e .` to install the project editable.
    * Use `pip install -e .[dev]` to install with development dependencies

## Usage

    from cpiets.cpi import CopyPasteImputation
    import pandas as pd

    cpi = CopyPasteImputation()
    data = pd.read_csv('data.csv')
    cpi.fit(data)
    result = cpi.impute()

### Data Requirements

**Example data:**

| time                | energy |
| ------------------- | ------:|
| 2012-01-02 00:15:00 |  11.60 |
| 2012-01-02 00:30:00 |  24.87 |
| 2012-01-02 00:45:00 |  37.31 |


The names of the columns are arbitrary.

**Supported time formats:**
* %Y-%m-%d %H:%M:%S (2020-01-17 13:37:42)
* %d-%b-%Y %H:%M:%S (17-Jan-2020 13:37:42)


## Example

To try CPI with simple test data, you can run the example

    python example/simple_imputation.py

and play around with the parameters.

The included data is derived from the [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) data set.