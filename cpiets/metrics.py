import numpy as np

def mape(actual_values, imputed_values) -> float:
    """Mean Absolute Percentage Error"""
    count = 0
    abs_percentage_error = 0
    for (act_value, imp_value) in zip(actual_values, imputed_values):
        if act_value != 0:
            abs_percentage_error += abs(act_value - imp_value) / act_value
            count += 1
    return abs_percentage_error / count

def wape(actual_values, imputed_values) -> float:
    """Weighted Absolute Percentage Error"""
    sum_abs_error = sum([abs(a - i) for (a, i) in zip(actual_values, imputed_values)])
    return sum_abs_error / sum(actual_values)
