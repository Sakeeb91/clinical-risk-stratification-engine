"""
Tests for data loading and synthetic data generation.

These are the first three tests to write as specified in the implementation plan.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSyntheticDataGenerator:
    """Tests for the synthetic EHR data generator."""

    def test_synthetic_data_has_correct_columns(self):
        """Verify synthetic data contains all required EHR columns."""
        from data.synthetic import generate_synthetic_ehr

        df = generate_synthetic_ehr(100)
        required_cols = ['patient_id', 'age', 'gender', 'readmission', 'sepsis', 'mortality']
        assert all(col in df.columns for col in required_cols), \
            f"Missing columns. Expected: {required_cols}, Got: {df.columns.tolist()}"

    def test_synthetic_data_has_correct_imbalance(self):
        """Verify synthetic data maintains realistic class imbalance."""
        from data.synthetic import generate_synthetic_ehr

        df = generate_synthetic_ehr(10000, random_state=42)
        readmission_rate = df['readmission'].mean()

        # Allow some variance but should be roughly 3-10%
        assert 0.03 < readmission_rate < 0.10, \
            f"Readmission rate {readmission_rate:.2%} outside expected range 3-10%"

    def test_synthetic_data_has_correct_size(self):
        """Verify synthetic data generates the requested number of patients."""
        from data.synthetic import generate_synthetic_ehr

        for n in [100, 500, 1000]:
            df = generate_synthetic_ehr(n)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_synthetic_data_reproducibility(self):
        """Verify same random_state produces identical data."""
        from data.synthetic import generate_synthetic_ehr

        df1 = generate_synthetic_ehr(100, random_state=42)
        df2 = generate_synthetic_ehr(100, random_state=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_data_age_range(self):
        """Verify ages are within realistic adult range."""
        from data.synthetic import generate_synthetic_ehr

        df = generate_synthetic_ehr(1000)

        assert df['age'].min() >= 18, "Age should be >= 18 (adult patients)"
        assert df['age'].max() <= 100, "Age should be <= 100"

    def test_synthetic_data_patient_ids_unique(self):
        """Verify all patient IDs are unique."""
        from data.synthetic import generate_synthetic_ehr

        df = generate_synthetic_ehr(1000)

        assert df['patient_id'].nunique() == len(df), "Patient IDs should be unique"


class TestDataValidator:
    """Tests for data validation (placeholder for Phase 1 implementation)."""

    @pytest.mark.skip(reason="Validator not yet implemented")
    def test_validator_catches_missing_required_field(self):
        """Verify validator raises error for missing required fields."""
        from data.validator import validate_ehr_data

        df = pd.DataFrame({'age': [50, 60]})  # Missing patient_id
        with pytest.raises(ValueError, match="missing required"):
            validate_ehr_data(df)

    @pytest.mark.skip(reason="Validator not yet implemented")
    def test_validator_catches_out_of_range_values(self):
        """Verify validator catches unrealistic values."""
        from data.validator import validate_ehr_data

        df = pd.DataFrame({
            'patient_id': ['P001'],
            'age': [150],  # Unrealistic age
        })
        with pytest.raises(ValueError, match="out of range"):
            validate_ehr_data(df)
