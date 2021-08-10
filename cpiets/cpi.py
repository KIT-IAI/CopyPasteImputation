"""
LGPL-3.0 License
Copyright (c) 2021 KIT-IAI Moritz Weber
"""

import os.path

from datetime import datetime, timedelta
from dateutil import parser
from fbprophet import Prophet
import numpy as np
import pandas as pd

from .utils import calculate_values_per_day, energy_to_power


class CopyPasteImputation():
    """An imputation method for energy time series."""

    def __init__(self, log_runtime: bool = False, rt_log_file: str = 'cpi_runtime.csv'):
        self.log_runtime = log_runtime
        self.rt_log_file = rt_log_file
        
        self.data = None
        self.vpd = None
        self.power = None
        self.daily_masks = None
        self.estimated_energy = None
        self.available_days = None

    def fit(self, ets: pd.DataFrame, values_per_day: int = None,
            starting_energy: float = 0.0):
        if self.log_runtime: timestamps = [datetime.now()]
        # linear interpolation of single missing values
        self.data = self._linear_interpolation(ets)
        
        if self.log_runtime: timestamps.append(datetime.now())

        # estimation of energy
        time = self.data.iloc[:, 0]
        if values_per_day is None:
            self.vpd = calculate_values_per_day(time)
        else:
            self.vpd = values_per_day

        energy = self.data.iloc[:, 1]
        self.power = energy_to_power(energy, starting_energy)

        energy_per_day = self._calc_energy_per_day(self.power)
        self.daily_masks = self._calc_daily_masks(self.power)

        weekly_pattern = self._estimate_weekly_pattern(
            time, energy_per_day, self.daily_masks)

        missing_energy = self._estimate_missing_energy_per_day(
            energy, self.power, weekly_pattern)
        self.estimated_energy = energy_per_day + missing_energy

        if self.log_runtime: timestamps.append(datetime.now())
        
        # compilation of list of available complete days
        self.available_days = self._compile_list_of_available_complete_days(
            time, energy_per_day, self.daily_masks)

        if self.log_runtime: 
            timestamps.append(datetime.now())
            self._log_runtimes(timestamps)
            

    def impute(self, w_weekday=1.0, w_season=5.0, w_energy=10.0,
            scaling_active=True) -> pd.Series:
        """Impute missing values in `ets` by copying and pasting fitting blocks into the gaps."""
        # calculation of dissimilarity between days
        # & copy and paste of best matching days
        imputed_power = self._impute_power(w_weekday, w_season, w_energy)
        
        if self.log_runtime: timestamps = [datetime.now()]
        if scaling_active:
            imputed_power = self._scale_imputation(imputed_power)
        
        if self.log_runtime: timestamps.append(datetime.now())
        imputed_energy = self._calculate_imputed_energy(imputed_power)

        if self.log_runtime: 
            timestamps.append(datetime.now())
            self._log_runtimes(timestamps)
        return imputed_energy

    def _impute_power(self, w_weekday, w_season, w_energy) -> pd.Series:
        time = self.data.iloc[:, 0]
        prediction = self.power.copy()
        power_na = self.power.isna()

        starting_date = parser.parse(time.iloc[0])
        starting_day = starting_date.timetuple().tm_yday

        min_e, max_e = self._calc_min_max_energy(self.available_days)
        distance = Distance(w_weekday, w_season, w_energy, min_e, max_e)

        matching_duration = timedelta(seconds=0)
        copy_paste_duration = timedelta(seconds=0)

        for i in range(self.daily_masks.shape[0]):
            if self.daily_masks[i] == 0:

                mstart = datetime.now()
                date = parser.parse(time.iloc[i * self.vpd])
                this_day = (
                    date.weekday(),
                    date.timetuple().tm_yday,
                    self.estimated_energy[i]
                )
                best_day = self._find_best_day(
                    distance, self.available_days, this_day)
                # index is day of year and starts with 1
                index = best_day[1]
                matching_duration += datetime.now() - mstart

                cpstart = datetime.now()
                best_day_data = self.power.iloc[(
                    index - starting_day)*self.vpd:(index - starting_day + 1)*self.vpd]

                for j in range(self.vpd):
                    if power_na.iloc[i*self.vpd + j]:
                        prediction.iloc[i*self.vpd + j] = best_day_data.iloc[j]
                copy_paste_duration += datetime.now() - cpstart

        if self.log_runtime:
            with open(self.rt_log_file, 'a') as log_file:
                log_file.write(f',{matching_duration.total_seconds()},{copy_paste_duration.total_seconds()}')

        return prediction

    def _linear_interpolation(self, ets: pd.DataFrame) -> pd.DataFrame:
        """Interpolate single missing values."""
        data = ets.copy()
        energy = data.iloc[:, 1]
        masks = energy.isna()

        for i in range(1, masks.shape[0] - 1):
            if masks.iloc[i] and not masks.iloc[i - 1] and not masks.iloc[i + 1]:
                data.iloc[i, 1] = (energy.iloc[i - 1] + energy.iloc[i + 1]) / 2

        return data

    def _calc_energy_per_day(self, power: pd.Series) -> np.array:
        power_na = power.isna()

        daily = np.zeros(power.shape[0] // self.vpd)
        count = 0
        for i in range(power.shape[0]):
            if not power_na[i]:
                daily[i // self.vpd] += power[i]
            else:
                count += 1
        return daily

    def _estimate_missing_energy_per_day(self, energy, power, weekly_pattern):
        power_na = power.isna()
        missing_energy = self._calc_average_missing_energy_per_day(
            energy, power_na)
        missing_energy = self._add_weekly_pattern_to_missing_energy(
            missing_energy, power_na, weekly_pattern)
        return missing_energy

    def _calc_average_missing_energy_per_day(self, energy, power_na):
        missing_energy = np.zeros(energy.shape[0] // self.vpd)

        missing_sequence_ongoing = False
        start = -1
        for i in range(energy.shape[0]):
            if power_na[i] and not missing_sequence_ongoing:
                # start missing sequence
                start = i
                missing_sequence_ongoing = True
            if not power_na[i] and missing_sequence_ongoing:
                # end missing sequence
                if start > 0:
                    energy_diff = energy[i - 1] - energy[start - 1]
                else:
                    energy_diff = energy[i - 1]
                avg = energy_diff / (i - start)
                for j in range(start, i):
                    missing_energy[j // self.vpd] += avg
                missing_sequence_ongoing = False
        return missing_energy

    def _add_weekly_pattern_to_missing_energy(self, missing_energy, power_na, weekly_pattern):
        missing_energy = missing_energy.copy()

        gap_start = -1
        sum_of_added_energy = 0
        for i in range(missing_energy.shape[0]):
            missing_values = sum(power_na[i*self.vpd:(i+1)*self.vpd])
            if missing_values > 0:  # this day is missing
                if gap_start == -1:
                    gap_start = i
                pattern_energy = weekly_pattern[i % 7]
                missing_energy[i] += pattern_energy
                sum_of_added_energy += pattern_energy
            else:  # this day is not missing
                if gap_start > -1:  # missing sequence ended
                    gap_end = i - 1
                    missing_energy = self._compensate_energy_of_missing_days(
                        missing_energy, sum_of_added_energy * -1, gap_start, gap_end)
                    sum_of_added_energy = 0
                    gap_start = -1

        return missing_energy

    def _compensate_energy_of_missing_days(self,
            missing_energy, energy_of_missing_days, start, end):
        missing_energy = missing_energy.copy()

        avg_energy = energy_of_missing_days / (end - start + 1)
        for j in range(start, end + 1):
            missing_energy[j] += avg_energy

        return missing_energy

    def _calc_daily_masks(self, power):
        power_na = power.isna()

        daily_masks = np.ones(power.shape[0] // self.vpd)
        for i in range(power.shape[0]):
            if power_na[i]:
                daily_masks[i // self.vpd] = 0.0
        return daily_masks

    def _estimate_weekly_pattern(self, time, energy_per_day, daily_masks):
        epd = energy_per_day.copy()

        for i in range(daily_masks.shape[0]):
            if daily_masks[i] < 1:
                epd[i] = np.nan

        daily_times = [i*self.vpd for i in range(daily_masks.shape[0])]
        try:
            time_index = pd.to_datetime(
                time.take(daily_times), format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            time_index = pd.to_datetime(
                time.take(daily_times), format='%d-%b-%Y %H:%M:%S')
        time_index.reset_index(drop=True, inplace=True)

        daily_data = pd.concat([time_index, pd.Series(
            epd)], keys=['ds', 'y'], axis=1)

        prophet_model = Prophet(weekly_seasonality=True,
                                yearly_seasonality=True)
        prophet_model.fit(daily_data)

        # `future` starts with the first day of the time series.
        # This allows us to select the corresponding pattern value
        # for day `i` by accessing `weekly[i % 7]`.
        future = prophet_model.make_future_dataframe(periods=0)
        forecast = prophet_model.predict(future)
        return forecast.get('weekly').head(7)

    def _compile_list_of_available_complete_days(self, time, daily_consumption, daily_masks):
        available_days = []
        for i in range(daily_consumption.shape[0]):
            if daily_masks[i] > 0:
                date = parser.parse(time.iloc[i * self.vpd])
                available_days.append((
                    date.weekday(),
                    date.timetuple().tm_yday,
                    daily_consumption[i]
                ))
        return available_days

    def _calc_min_max_energy(self, available_days):
        min_e = min(available_days, key=lambda x: x[2])[2]
        max_e = max(available_days, key=lambda x: x[2])[2]
        return min_e, max_e

    def _find_best_day(self, distance, available_days, day):
        distances = []
        for d in available_days:
            distances.append(distance.distance(day, d))
        min_index = np.argmin(distances)
        return available_days[min_index]

    def _scale_imputation(self, imp_power):
        energy = self.data.iloc[:, 1]
        scaled_power = imp_power.copy()
        power_na = self.power.isna()

        start = -1
        for i in range(imp_power.shape[0]):
            if power_na.iloc[i]:
                if start < 0:
                    start = i
            else:
                if start > -1:  # end of gap
                    energy_gt = energy.iloc[i - 1] - energy.iloc[start - 1]
                    energy_imp = imp_power.iloc[start:i].sum()
                    factor = energy_gt / energy_imp
                    scaled_power.iloc[start:i] = imp_power.iloc[start:i] * factor
                start = -1

        return scaled_power

    def _calculate_imputed_energy(self, imputed_power):
        energy = self.data.iloc[:, 1]
        imputed_energy = energy.copy()
        energy_na = energy.isna()
        for i in range(energy.shape[0]):
            if energy_na.iloc[i]:
                imputed_energy.iloc[i] = imputed_energy.iloc[i - 1] \
                    + imputed_power.iloc[i]
        return imputed_energy

    def _log_runtimes(self, timestamps):
        with open(self.rt_log_file, 'a') as log_file:
            for i in range(1, len(timestamps)):
                duration = timestamps[i] - timestamps[i - 1]
                log_file.write(f',{duration.total_seconds()}')


class Distance:
    def __init__(self, w_weekday, w_season, w_energy, min_e, max_e):
        self.w_weekday = w_weekday
        self.w_season = w_season
        self.w_energy = w_energy

        self.min_e = min_e
        self.max_e = max_e

    def distance(self, day1, day2):
        weekday_distance = self._weekday_distance(day1[0], day2[0])
        yearday_distance = self._yearday_distance(day1[1], day2[1])
        energy_distance = self._energy_distance(day1[2], day2[2])

        return self.w_weekday * weekday_distance \
            + self.w_season * yearday_distance \
            + self.w_energy * energy_distance

    def _weekday_distance(self, weekday_a, weekday_b):
        if weekday_a == weekday_b:
            return 0.0
        if weekday_a in range(5) and weekday_b in [5, 6] \
                or weekday_a in [5, 6] and weekday_b in range(5):
            return 1.0
        return 0.5

    def _yearday_distance(self, yearday_a, yearday_b):
        days = abs(yearday_a - yearday_b)
        if days > 182:
            days = 365 - days
        return days / 182

    def _energy_distance(self, energy_a, energy_b):
        return abs(energy_a - energy_b) / (self.max_e - self.min_e)
