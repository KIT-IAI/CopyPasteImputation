import matplotlib.pyplot as plt
import pandas as pd

from cpiets.cpi import CopyPasteImputation
from cpiets.utils import energy_to_power, estimate_starting_energy


def main():
    # set values per day (96 = one value every 15 minutes)
    vpd = 96

    # read the time series data from the example file
    # office_96: 2 weeks of typical office energy time series
    # office_96_feb: same time series, but starting in February
    # office_96+500: same time series, but with a constant 500 kWh added
    # office_96_complete: no missing values
    timeseries = pd.read_csv('example/data/office_96.csv')

    # starting energy is required to calculate a complete power time series
    starting_energy = estimate_starting_energy(timeseries.iloc[:, 1])
    print(f'starting energy: {starting_energy}')

    # create an instance of the imputation algorithm
    # try changing the weight parameters to achieve different results
    # scaling of the imputed values can be turned on or off (defaults to on)
    cpi = CopyPasteImputation(w_energy=5.0, w_season=10.0, w_weekday=1.0, scaling_active=True)

    # impute the missing values in the time series
    ## cpi can actually figure out the values per day automatically; try removing the parameter
    result = cpi.impute(ets=timeseries, values_per_day=vpd, starting_energy=starting_energy)

    # plot the resulting energy time series with imputed values in blue
    plt.plot(result)
    plt.plot(timeseries.iloc[:, 1])
    plt.show()

    # plot the power time series derived from the imputed energy time series
    plt.plot(energy_to_power(result, starting_energy, vpd / 24.0))
    plt.plot(energy_to_power(timeseries.iloc[:, 1], starting_energy, vpd / 24.0))
    plt.show()

if __name__ == "__main__":
    main()
