import pandas as pd

from forgepulse.drift import population_stability_index


def test_identical_distributions_have_near_zero_psi():
    values = pd.Series(range(1, 101), dtype=float)
    assert population_stability_index(values, values) < 1e-9


def test_shifted_distribution_has_positive_psi():
    reference = pd.Series(range(1, 101), dtype=float)
    current = reference + 25
    assert population_stability_index(reference, current) > 0
