from datetime import timedelta
from dateutil import parser
import numpy as np
import pandas as pd

def calculate_values_per_day(time: pd.Series) -> int:
    """Calculate the values per day from a given series of time strings."""
    first = parser.parse(time.iloc[0])
    second = parser.parse(time.iloc[1])

    delta = second - first
    one_day = timedelta(hours=24)

    return int(one_day / delta)


def estimate_starting_energy(energy: pd.Series, samples: int = 5) -> float:
    """Estimate the energy of a time series before the first value.  
    This is necessary to compute the complete power time series.
    """
    avg_difference = 0.0
    samples = min(samples, energy.shape[0])

    for i in range(samples):
        avg_difference += energy.iloc[i + 1] - energy.iloc[i]

    avg_difference /= samples

    return max(0.0, energy.iloc[0] - avg_difference)


def energy_to_power(energy: pd.Series, starting_energy: float, time_factor: int = 1.0) -> pd.Series:
    """Derive a power time series from an energy time series."""
    power = []
    power.append((energy.iloc[0] - starting_energy) * time_factor)

    energy_na = energy.isna()
    for i in range(1, energy.shape[0]):
        if not energy_na.iloc[i] and not energy_na.iloc[i - 1]:
            power.append(
                (energy.iloc[i] - energy.iloc[i - 1]) * time_factor)
        else:
            power.append(np.nan)

    return pd.Series(power)