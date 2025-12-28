# Clinical Risk Stratification Engine

A patient risk stratification system that predicts multiple adverse outcomes from structured Electronic Health Record (EHR) data. The system provides calibrated probability estimates and generates clinician-interpretable explanations for each prediction while ensuring fairness across demographic subgroups.

## Overview

This engine addresses three critical clinical prediction tasks:
- **30-Day Readmission Risk**: Predicting likelihood of hospital readmission within 30 days of discharge
- **Sepsis Onset Prediction**: Early warning system for sepsis development in hospitalized patients
- **Mortality Risk Assessment**: Estimating in-hospital or short-term mortality probability

## Key Features

- **Multi-Output Classification**: Shared feature representations across prediction targets for improved generalization
- **Class Imbalance Handling**: SMOTE variants and cost-sensitive learning to address extreme outcome imbalances
- **Probability Calibration**: Isotonic regression ensuring reliable probability estimates for clinical decision-making
- **Explainable Predictions**: SHAP (SHapley Additive exPlanations) values integrated into the prediction pipeline
- **Clinical Feature Selection**: Recursive feature elimination with domain-specific constraints
- **Fairness Auditing**: Disparate impact metrics across demographic subgroups to detect and mitigate bias

## Project Structure

```
clinical-risk-stratification-engine/
├── src/
│   ├── data/                 # Data loading and preprocessing
│   ├── features/             # Feature engineering and selection
│   ├── models/               # Model training and prediction
│   ├── calibration/          # Probability calibration
│   ├── explainability/       # SHAP integration
│   └── fairness/             # Bias detection and mitigation
├── tests/                    # Unit and integration tests
├── notebooks/                # Exploratory analysis and demos
├── docs/                     # Documentation and implementation plan
├── configs/                  # Configuration files
└── data/                     # Data directory (not tracked)
    ├── raw/
    ├── processed/
    └── synthetic/
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/clinical-risk-stratification-engine.git
cd clinical-risk-stratification-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.risk_engine import ClinicalRiskEngine

# Initialize the engine
engine = ClinicalRiskEngine()

# Load and preprocess data
engine.fit(X_train, y_train)

# Generate predictions with explanations
predictions = engine.predict_proba(X_test)
explanations = engine.explain(X_test)

# Audit for fairness
fairness_report = engine.audit_fairness(X_test, sensitive_features)
```

## Requirements

- Python 3.9+
- scikit-learn
- imbalanced-learn
- shap
- pandas
- numpy

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)

## Data Requirements

This system is designed to work with structured EHR data containing:
- Patient demographics
- Vital signs
- Laboratory results
- Diagnosis codes (ICD-10)
- Procedure codes
- Medication history
- Prior utilization history

Note: Due to healthcare data privacy regulations (HIPAA), no real patient data is included. The project uses synthetic data for development and testing.

## License

MIT License

## Contributing

Contributions are welcome. Please read the contributing guidelines before submitting pull requests.

## Disclaimer

This tool is intended for research and educational purposes. It should not be used as a sole basis for clinical decision-making without proper validation and regulatory approval.
