"""
Synthetic EHR Data Generator

This module generates realistic synthetic Electronic Health Record data
for testing the clinical risk stratification engine.

Learning Context:
- We use numpy's random generator for reproducibility
- Class imbalance is set to match real-world clinical data
- Features are generated with realistic correlations (e.g., age correlates with mortality)
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_ehr(
    n_patients: int = 1000,
    random_state: int = 42,
    readmission_rate: float = 0.05,
    sepsis_rate: float = 0.02,
    mortality_rate: float = 0.03
) -> pd.DataFrame:
    """
    Generate synthetic Electronic Health Record data for testing.

    Parameters
    ----------
    n_patients : int
        Number of patient records to generate
    random_state : int
        Random seed for reproducibility
    readmission_rate : float
        Target 30-day readmission rate (default 5%)
    sepsis_rate : float
        Target sepsis onset rate (default 2%)
    mortality_rate : float
        Target mortality rate (default 3%)

    Returns
    -------
    pd.DataFrame
        Synthetic EHR data with patient demographics, vitals, labs, and outcomes

    Examples
    --------
    >>> df = generate_synthetic_ehr(1000)
    >>> df.shape
    (1000, ...)
    >>> df['readmission'].mean()  # Should be ~0.05
    """
    rng = np.random.default_rng(random_state)

    # Generate patient IDs
    patient_ids = [f"P{str(i).zfill(6)}" for i in range(n_patients)]

    # Demographics
    age = rng.normal(65, 15, n_patients).clip(18, 100).astype(int)
    gender = rng.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
    race = rng.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
        n_patients,
        p=[0.60, 0.15, 0.15, 0.05, 0.05]
    )

    # TODO: Add vital signs (heart rate, blood pressure, temperature, respiratory rate)
    # TODO: Add laboratory values (WBC, creatinine, lactate, etc.)
    # TODO: Add admission information (admission type, length of stay prior)
    # TODO: Generate correlated outcomes based on risk factors

    # Placeholder outcomes (to be replaced with correlated generation)
    readmission = rng.choice([0, 1], n_patients, p=[1-readmission_rate, readmission_rate])
    sepsis = rng.choice([0, 1], n_patients, p=[1-sepsis_rate, sepsis_rate])
    mortality = rng.choice([0, 1], n_patients, p=[1-mortality_rate, mortality_rate])

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'gender': gender,
        'race': race,
        'readmission': readmission,
        'sepsis': sepsis,
        'mortality': mortality,
    })

    return df


if __name__ == "__main__":
    # Quick test
    df = generate_synthetic_ehr(1000)
    print(f"Generated {len(df)} patients")
    print(f"Readmission rate: {df['readmission'].mean():.2%}")
    print(f"Sepsis rate: {df['sepsis'].mean():.2%}")
    print(f"Mortality rate: {df['mortality'].mean():.2%}")
    print(df.head())
