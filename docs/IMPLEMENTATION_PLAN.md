# Clinical Risk Stratification Engine - Implementation Plan

## Expert Role: Healthcare ML Engineer

**Why this role is optimal:** This project sits at the intersection of machine learning, healthcare domain knowledge, and regulatory compliance. A Healthcare ML Engineer understands:
- Clinical workflow constraints (predictions must be fast, interpretable)
- Regulatory requirements (FDA guidelines for clinical decision support)
- The critical importance of calibrated probabilities in healthcare
- Class imbalance realities in adverse outcome prediction
- Fairness considerations when models affect patient care

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLINICAL RISK STRATIFICATION ENGINE                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   RAW EHR    │────▶│    DATA      │────▶│   FEATURE    │────▶│   FEATURE    │
│    DATA      │     │  VALIDATOR   │     │  EXTRACTOR   │     │   SELECTOR   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                    │                     │
                            ▼                    ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                     │   Missing    │     │  Numerical   │     │  Recursive   │
                     │   Value      │     │  Encoding    │     │  Feature     │
                     │   Handler    │     │  Scaling     │     │  Elimination │
                     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-OUTPUT CLASSIFIER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SHARED FEATURE REPRESENTATION                     │    │
│  │                         (Hidden Layers)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│            │                      │                      │                   │
│            ▼                      ▼                      ▼                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   READMISSION    │  │     SEPSIS       │  │   MORTALITY      │          │
│  │     HEAD         │  │     HEAD         │  │     HEAD         │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION LAYER                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    ISOTONIC      │  │    ISOTONIC      │  │    ISOTONIC      │          │
│  │   REGRESSION     │  │   REGRESSION     │  │   REGRESSION     │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXPLAINABILITY MODULE                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SHAP EXPLAINER                                  │    │
│  │     ┌───────────┐    ┌───────────┐    ┌───────────┐                 │    │
│  │     │  Feature  │    │   Force   │    │  Summary  │                 │    │
│  │     │   Values  │    │   Plots   │    │   Plots   │                 │    │
│  │     └───────────┘    └───────────┘    └───────────┘                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FAIRNESS AUDITOR                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Demographic │  │  Equalized   │  │  Disparate   │  │  Calibration │    │
│  │    Parity    │  │    Odds      │  │   Impact     │  │   by Group   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  PREDICTION API  │
                        │   - Probabilities│
                        │   - Explanations │
                        │   - Fairness     │
                        └──────────────────┘
```

### Data Flow Summary

1. **Ingestion**: Raw EHR data enters the system
2. **Validation**: Schema and value range checks
3. **Preprocessing**: Missing value imputation, encoding, scaling
4. **Feature Selection**: RFE with clinical domain constraints
5. **Prediction**: Multi-output model with shared representations
6. **Calibration**: Isotonic regression per output
7. **Explanation**: SHAP values for each prediction
8. **Fairness Audit**: Metrics across demographic groups

---

## Technology Selection

| Component | Technology | Rationale | Tradeoffs | Fallback |
|-----------|------------|-----------|-----------|----------|
| **Core ML** | scikit-learn | Well-documented, junior-friendly, sufficient for tabular data | Less flexible than PyTorch for complex architectures | XGBoost for gradient boosting |
| **Imbalance** | imbalanced-learn | Integrates with sklearn, multiple SMOTE variants | Memory-intensive for large datasets | Class weights in sklearn |
| **Calibration** | sklearn.calibration | Built-in isotonic regression | Limited to binary classification | Manual calibration mapping |
| **Explainability** | SHAP | Gold standard for feature attribution, clinician-friendly | Slow for large datasets | LIME (faster but less stable) |
| **Fairness** | fairlearn | Microsoft-backed, good documentation | Limited mitigation algorithms | Manual metric computation |
| **Data** | pandas | Universal tabular data library | Memory limits with large data | Dask for out-of-core |
| **Testing** | pytest | Standard Python testing | None | unittest |
| **Validation** | pydantic | Type-safe data validation | Learning curve | Manual validation |

### Learning Context Notes

**For Junior Developer:**
- Start with sklearn's documentation tutorials before diving into code
- SHAP has excellent visual tutorials - watch the introductory video
- Understand the difference between stratified and regular train/test splits (critical for imbalanced data)
- Learn what "calibration" means in probability context before implementing

---

## Phased Implementation Plan

### Phase 1: Data Foundation

**Objective:** Create robust data loading, validation, and synthetic data generation pipeline.

**Scope:**
- `src/data/__init__.py`
- `src/data/loader.py` - EHR data loading utilities
- `src/data/validator.py` - Schema and value validation
- `src/data/synthetic.py` - Synthetic data generator for testing
- `configs/schema.yaml` - Data schema definition

**Deliverables:**
1. Synthetic dataset generator producing 10,000 patient records
2. Data validator catching 5 types of data quality issues
3. Data loader supporting CSV and Parquet formats

**Verification:**
```bash
python -m pytest tests/test_data.py -v
python -c "from src.data.synthetic import generate_synthetic_ehr; df = generate_synthetic_ehr(1000); print(df.shape)"
```

**Technical Challenges:**
- Generating realistic correlations between clinical variables
- Maintaining class imbalance ratios typical of real data (~5% readmission, ~2% sepsis, ~3% mortality)

**Definition of Done:**
- [ ] Synthetic data generator creates valid EHR-like data
- [ ] Validator catches missing required fields
- [ ] Validator catches out-of-range values
- [ ] All unit tests pass
- [ ] Code has docstrings explaining clinical context

**Contingency:** If synthetic data generation is complex, use simple random generation with manual imbalance injection.

---

### Phase 2: Feature Engineering Pipeline

**Objective:** Build feature extraction and selection pipeline with clinical constraints.

**Scope:**
- `src/features/__init__.py`
- `src/features/extractor.py` - Feature extraction from raw data
- `src/features/preprocessor.py` - Missing value handling, encoding, scaling
- `src/features/selector.py` - RFE with clinical constraints
- `configs/clinical_constraints.yaml` - Features that must be included/excluded

**Deliverables:**
1. Preprocessor handling 3 missing value strategies (mean, median, indicator)
2. Feature extractor producing 50+ features from raw EHR
3. Selector respecting clinical must-include constraints

**Verification:**
```bash
python -m pytest tests/test_features.py -v
python -c "from src.features.extractor import FeatureExtractor; print(FeatureExtractor().get_feature_names())"
```

**Technical Challenges:**
- Encoding high-cardinality categorical variables (ICD codes)
- Handling temporal features (time since admission)
- Respecting clinical constraints during RFE

**Definition of Done:**
- [ ] Preprocessor handles missing values without data leakage
- [ ] Feature extractor produces consistent feature set
- [ ] Selector respects must-include constraints
- [ ] Pipeline is sklearn-compatible (fit/transform)
- [ ] All unit tests pass

**Contingency:** If ICD code handling is complex, start with top-100 codes only.

---

### Phase 3: Multi-Output Model Core

**Objective:** Implement multi-output classifier with shared representations and imbalance handling.

**Scope:**
- `src/models/__init__.py`
- `src/models/base.py` - Base model class
- `src/models/multi_output.py` - Multi-output classifier wrapper
- `src/models/imbalance.py` - SMOTE and cost-sensitive utilities

**Deliverables:**
1. Multi-output classifier predicting all 3 outcomes
2. SMOTE-NC integration for mixed feature types
3. Cost-sensitive learning wrapper

**Verification:**
```bash
python -m pytest tests/test_models.py -v
python -c "from src.models.multi_output import MultiOutputRiskModel; m = MultiOutputRiskModel(); print(m)"
```

**Technical Challenges:**
- Applying SMOTE correctly to multi-output (resample per target vs jointly)
- Choosing appropriate base classifier (RF vs XGBoost vs Logistic)
- Avoiding data leakage with resampling in cross-validation

**Definition of Done:**
- [ ] Model trains on synthetic data without errors
- [ ] SMOTE applied correctly (after train/test split)
- [ ] Predictions return 3 probability outputs
- [ ] Cross-validation runs without data leakage
- [ ] All unit tests pass

**Contingency:** Start with 3 separate models if multi-output wrapper is complex.

---

### Phase 4: Probability Calibration

**Objective:** Implement isotonic regression calibration for reliable probability estimates.

**Scope:**
- `src/calibration/__init__.py`
- `src/calibration/isotonic.py` - Isotonic calibration wrapper
- `src/calibration/metrics.py` - Calibration metrics (Brier score, reliability diagrams)

**Deliverables:**
1. Post-hoc calibration for each output
2. Calibration evaluation metrics
3. Reliability diagram generator

**Verification:**
```bash
python -m pytest tests/test_calibration.py -v
python -c "from src.calibration.metrics import brier_score_multi; print(brier_score_multi.__doc__)"
```

**Technical Challenges:**
- Calibration on held-out data (not training data)
- Maintaining calibration across different operating points
- Handling very small minority class probabilities

**Definition of Done:**
- [ ] Calibrated probabilities improve Brier score
- [ ] Reliability diagrams show improved calibration
- [ ] Calibration works for all 3 outputs
- [ ] All unit tests pass

**Contingency:** Use Platt scaling if isotonic is unstable for small datasets.

---

### Phase 5: Explainability Integration

**Objective:** Integrate SHAP explanations into the prediction pipeline.

**Scope:**
- `src/explainability/__init__.py`
- `src/explainability/shap_explainer.py` - SHAP wrapper
- `src/explainability/visualizations.py` - Clinician-friendly visualizations

**Deliverables:**
1. SHAP explainer for each prediction
2. Feature importance rankings
3. Force plot generation for individual patients

**Verification:**
```bash
python -m pytest tests/test_explainability.py -v
python -c "from src.explainability.shap_explainer import RiskExplainer; print(RiskExplainer.__doc__)"
```

**Technical Challenges:**
- SHAP computation time for tree-based models
- Explaining calibrated vs uncalibrated predictions
- Generating useful visualizations for multi-output

**Definition of Done:**
- [ ] SHAP values computed for all features
- [ ] Force plots render correctly
- [ ] Explanation time < 1 second per patient
- [ ] All unit tests pass

**Contingency:** Use TreeExplainer (fast) before trying KernelExplainer (slow but universal).

---

### Phase 6: Fairness Auditing

**Objective:** Implement fairness metrics and bias detection across demographic groups.

**Scope:**
- `src/fairness/__init__.py`
- `src/fairness/metrics.py` - Fairness metric computations
- `src/fairness/auditor.py` - Comprehensive fairness auditor
- `src/fairness/report.py` - Fairness report generator

**Deliverables:**
1. Disparate impact ratio computation
2. Equalized odds difference
3. Calibration comparison across groups
4. Fairness audit report

**Verification:**
```bash
python -m pytest tests/test_fairness.py -v
python -c "from src.fairness.auditor import FairnessAuditor; print(FairnessAuditor.__doc__)"
```

**Technical Challenges:**
- Choosing appropriate fairness metric for healthcare context
- Handling intersectional groups (race + gender)
- Balancing accuracy vs fairness tradeoffs

**Definition of Done:**
- [ ] Disparate impact computed correctly
- [ ] Audit report generated as DataFrame
- [ ] Handles missing demographic data gracefully
- [ ] All unit tests pass

**Contingency:** Start with binary sensitive attributes before intersectional.

---

### Phase 7: Integration and API

**Objective:** Integrate all components into a unified prediction API.

**Scope:**
- `src/models/risk_engine.py` - Main engine class
- `src/api/__init__.py`
- `src/api/predict.py` - Prediction endpoint logic
- `notebooks/demo.ipynb` - End-to-end demonstration

**Deliverables:**
1. Unified `ClinicalRiskEngine` class
2. Single-patient and batch prediction methods
3. Demo notebook with full pipeline

**Verification:**
```bash
python -m pytest tests/test_integration.py -v
jupyter nbconvert --execute notebooks/demo.ipynb
```

**Technical Challenges:**
- Ensuring consistent preprocessing at inference time
- Handling missing features at prediction time
- Serialization/deserialization of fitted pipeline

**Definition of Done:**
- [ ] Engine trains, predicts, explains, and audits
- [ ] Demo notebook runs end-to-end
- [ ] Pipeline serializes to disk correctly
- [ ] All integration tests pass

**Contingency:** Simplify to single-patient API if batch is complex.

---

## Risk Assessment

| Risk | Likelihood | Impact | L×I | Early Warning Signs | Mitigation |
|------|------------|--------|-----|---------------------|------------|
| SHAP computation too slow | High | Medium | 🔴 | >5s per patient | Use TreeExplainer, sample background data |
| Synthetic data not realistic | Medium | High | 🔴 | Models achieve unrealistic accuracy | Add noise, verify with domain expert |
| Calibration instability | Medium | Medium | 🟡 | Brier score worse after calibration | Use larger calibration set, try Platt |
| Multi-output correlations ignored | Low | Medium | 🟡 | Outputs are independent | Add explicit correlation modeling |
| Fairness metrics conflict | Medium | Low | 🟢 | No single "fair" solution | Report multiple metrics, document tradeoffs |
| Memory issues with SMOTE | Low | High | 🟡 | OOM errors | Use SMOTE-ENN, reduce oversampling ratio |

---

## Testing Strategy

### Testing Framework: pytest

### Test Categories

1. **Unit Tests** (`tests/test_*.py`)
   - Each module has corresponding test file
   - Minimum 80% code coverage target
   - Mock external dependencies

2. **Integration Tests** (`tests/test_integration.py`)
   - End-to-end pipeline tests
   - Test with synthetic data

3. **Validation Tests** (`tests/test_validation.py`)
   - Clinical validity checks
   - Range checks on outputs

### First Three Tests to Write

```python
# tests/test_data.py

def test_synthetic_data_has_correct_columns():
    """Verify synthetic data contains all required EHR columns."""
    from src.data.synthetic import generate_synthetic_ehr
    df = generate_synthetic_ehr(100)
    required_cols = ['patient_id', 'age', 'gender', 'readmission', 'sepsis', 'mortality']
    assert all(col in df.columns for col in required_cols)

def test_synthetic_data_has_correct_imbalance():
    """Verify synthetic data maintains realistic class imbalance."""
    from src.data.synthetic import generate_synthetic_ehr
    df = generate_synthetic_ehr(10000)
    readmission_rate = df['readmission'].mean()
    assert 0.03 < readmission_rate < 0.10  # 3-10% is realistic

def test_validator_catches_missing_required_field():
    """Verify validator raises error for missing required fields."""
    from src.data.validator import validate_ehr_data
    import pandas as pd
    df = pd.DataFrame({'age': [50, 60]})  # Missing patient_id
    with pytest.raises(ValueError, match="missing required"):
        validate_ehr_data(df)
```

---

## First Concrete Task

### File to Create: `src/data/synthetic.py`

### Function Signature:
```python
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
    """
```

### Starter Code:

```python
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
```

### Verification Method:
```bash
cd clinical-risk-stratification-engine
python src/data/synthetic.py
```

### First Commit Message:
```
Add synthetic EHR data generator skeleton

Implements basic structure for generating synthetic Electronic Health
Record data with configurable class imbalance rates. Includes patient
demographics (age, gender, race) and outcome labels.

TODOs remain for vital signs, laboratory values, and correlated
outcome generation.
```

---

## Summary

This implementation plan provides a clear path from empty repository to functional clinical risk stratification engine. Each phase builds on the previous, and the project remains useful even if stopped at any phase boundary.

**Total Phases:** 7
**Estimated Complexity:** Medium-High (but manageable with phased approach)
**Key Dependencies:** scikit-learn, imbalanced-learn, shap, fairlearn, pandas

**Next Step:** Implement Phase 1 starting with `src/data/synthetic.py`
