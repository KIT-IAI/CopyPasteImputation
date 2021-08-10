"""
LGPL-3.0 License
Copyright (c) 2021 KIT-IAI Moritz Weber
"""

from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from .cpi import CopyPasteImputation as CPI
from .metrics import mape, wape
from .utils import energy_to_power, sum_per_gap, mask_values

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def calibrate_cpi(dataset: Dict[str, pd.DataFrame], grid: Dict[str, List[int]], log_results = False, log_file_name: str = None, thread_count: int = 4) -> Dict:
    """Determine best weight combination in [grid] for [dataset].

    
    dataset -- dict of time series; time series are expected as DataFrame with columns: time, energy (with NaN), energy (without NaN)
    grid -- dict with list of values for every weight; expected keys: w_weekday, w_season, w_energy
    """

    wws = grid['w_weekday']
    wss = grid['w_season']
    wes = grid['w_energy']

    input_data = [(wws, wss, wes, x) for x in dataset.items()]

    with Pool(thread_count) as p:
        results = p.starmap(eval_weights_on_timeseries, input_data)

    all_results = pd.concat(results, ignore_index=True)
    
    if log_results:
        if log_file_name is None:
            log_file_name = f'cpi_calibration_{datetime.now()}.csv'
        all_results.to_csv(log_file_name, index=False)

    min_mape_p = all_results['mape_p'].min()
    max_mape_p = all_results['mape_p'].max()
    if min_mape_p < max_mape_p:
        all_results['mape_p'] = (all_results['mape_p'] - min_mape_p) / (max_mape_p - min_mape_p)

    min_wape_e = all_results['wape_e'].min()
    max_wape_e = all_results['wape_e'].max()
    if min_wape_e < max_wape_e:
        all_results['wape_e'] = (all_results['wape_e'] - min_wape_e) / (max_wape_e - min_wape_e)

    mean_grouped = all_results.groupby(['ww', 'ws', 'we']).mean()
    mean_grouped['error_sum'] = mean_grouped['mape_p'] + mean_grouped['wape_e']
    mean_weights = mean_grouped['error_sum'].idxmin()

    median_grouped = all_results.groupby(['ww', 'ws', 'we']).median()
    median_grouped['error_sum'] = median_grouped['mape_p'] + median_grouped['wape_e']
    median_weights = median_grouped['error_sum'].idxmin()

    return {
        'info': '(w_weekday, w_season, w_energy)',
        'mean': mean_weights,
        'median': median_weights,
    }

def eval_weights_on_timeseries(wws, wss, wes, ts):
    name, data = ts
    ets = data.iloc[:, 0:2]

    power_gaps = energy_to_power(data.iloc[:, 1], 0.0, 4.0)

    actual_energy = data.iloc[:, 2]
    actual_power = energy_to_power(actual_energy, 0.0, 4.0)

    masks = np.ones(len(actual_energy))
    masks[np.where(data.iloc[:, 1].isna())] = 0.0

    power_masks = np.ones(len(power_gaps))
    power_masks[np.where(power_gaps.isna())] = 0.0

    cpi = CPI()
    cpi.fit(ets)

    results = pd.DataFrame(np.zeros((len(wws)*len(wss)*len(wes), 6)), columns=['ww', 'ws', 'we', 'ts', 'mape_p', 'wape_e'])

    i = 0
    for ww in wws:
        for ws in wss:
            for we in wes:
                imputed_energy = cpi.impute(ww, ws, we, scaling_active=True)
                imputed_power = energy_to_power(imputed_energy, 0.0, 4.0)

                energies_a = sum_per_gap(actual_power, masks)
                energies_i = sum_per_gap(imputed_power, masks)
                wape_e = wape(energies_a, energies_i)

                actual_power_m = mask_values(actual_power, power_masks)
                imputed_power_m = mask_values(imputed_power, power_masks)
                mape_p = mape(actual_power_m, imputed_power_m)

                results.iloc[i, :] = [ww, ws, we, name, mape_p, wape_e]
                i += 1
    return results
