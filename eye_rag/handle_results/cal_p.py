import numpy as np
from scipy import stats


def calculate_spearman_extended(ranking_a, ranking_b, ranking_c=None):
    """
    Calculates the Spearman's rank correlation coefficient and its p-value.

    Args:
        ranking_a: A list of numbers representing the first ranking.
        ranking_b: A list of numbers representing the second ranking.
        ranking_c: Optional. A list of numbers representing the third ranking.

    Returns:
        If ranking_c is None: A tuple containing the Spearman's rho and the p-value.
        If ranking_c is provided: A dictionary with pairwise correlations, p-values, and averages.
    """
    if ranking_c is None:
        rho, p_value = stats.spearmanr(ranking_a, ranking_b)
        return rho, p_value
    else:
        rho_ab, p_ab = stats.spearmanr(ranking_a, ranking_b)
        rho_ac, p_ac = stats.spearmanr(ranking_a, ranking_c)
        rho_bc, p_bc = stats.spearmanr(ranking_b, ranking_c)

        avg_rho = (rho_ab + rho_ac + rho_bc) / 3
        avg_p = (p_ab + p_ac + p_bc) / 3

        return {
            'pairwise': {
                'a_b': {'rho': rho_ab, 'p_value': p_ab},
                'a_c': {'rho': rho_ac, 'p_value': p_ac},
                'b_c': {'rho': rho_bc, 'p_value': p_bc}
            },
            'average': {
                'rho': avg_rho,
                'p_value': avg_p
            }
        }


def calculate_friedman_p_value(ranking_data):
    """
    Calculates the p-value for multiple groups of ranking results using the Friedman test.

    The Friedman test is a non-parametric statistical test used to detect differences in
    treatments across multiple test attempts. It is used for ranked data.

    Args:
        ranking_data (list of lists or np.ndarray): A 2D array-like structure where:
            - Each row represents a "block" (e.g., a setting, a judge, a trial).
            - Each column represents a "group" or "treatment" being ranked (e.g., a method).
            - The cells contain the ranks (e.g., 1, 2, 3, ...).

    Returns:
        tuple: A tuple containing:
            - statistic (float): The Friedman test statistic.
            - p_value (float): The associated p-value.

    Raises:
        ValueError: If the input data is not a 2D array or is empty.
    """
    if not hasattr(ranking_data, '__len__') or not hasattr(ranking_data[0], '__len__'):
        raise ValueError("Input 'ranking_data' must be a 2D array-like structure.")

    data = np.asarray(ranking_data)

    if data.ndim != 2 or data.size == 0:
        raise ValueError("Input 'ranking_data' must be a non-empty 2D array.")

    num_groups = data.shape[1]

    if num_groups < 2:
        print("Warning: Friedman test requires at least 2 groups to compare. Returning None.")
        return None, None

    statistic, p_value = stats.friedmanchisquare(*data.T)

    return statistic, p_value
