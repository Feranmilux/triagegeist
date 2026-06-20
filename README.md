# Triagegeist: Emergency Department Triage Acuity Prediction
 
**Predicting ESI-level triage severity from structured patient intake data using machine learning.**
 
Submitted to the Laitinen-Fredriksson Foundation Triagegeist competition.
 
---
 
## Overview
 
This repository contains a machine learning pipeline to predict Emergency Severity Index (ESI) triage acuity levels from structured patient intake data. The model is trained on publicly available NHAMCS (National Hospital Ambulatory Medical Care Survey) data and uses XGBoost classification with rigorous evaluation and clinical validation.
 
**Key Files:**
- `triagegeist_notebook.ipynb` — Complete analysis notebook (Kaggle-hosted)
- `train.py` — Standalone training script
- `requirements.txt` — Python dependencies
- `data/` — NHAMCS preprocessing scripts (optional local setup)
---
 
## Quick Start
 
### Prerequisites
- Python 3.9+
- Conda or pip
### Setup (5 minutes)
 
```bash
# Clone repo
git clone https://github.com/Feranmilux/triagegeist.git
cd triagegeist
 
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
 
# Install dependencies
pip install -r requirements.txt
```
 
### Run the Notebook Locally
 
```bash
jupyter notebook triagegeist_notebook.ipynb
```
 
The notebook will:
1. Fetch NHAMCS 2022 data from CDC
2. Preprocess and engineer features
3. Train XGBoost model
4. Evaluate on test set
5. Generate clinical insights
**Expected runtime:** ~8 minutes (first run may take longer for data download)
 
### Expected Output
- Model accuracy: ~72-75% (test set)
- Feature importance plots
- Confusion matrix
- Per-class performance metrics
---
 
## Data
 
### Source
**NHAMCS (2022)** — National Hospital Ambulatory Medical Care Survey
- Public dataset from CDC/NCHS
- ~400,000 ED visits across United States
- Variables: vitals, demographics, chief complaint, triage severity, disposition
**Citation:**
National Center for Health Statistics. (2023). NHAMCS—National Hospital Ambulatory Medical Care Survey [Data]. CDC. https://www.cdc.gov/nchs/ahcd/
 
### Feature Set
- **Demographics:** Age, sex
- **Vitals:** Systolic/diastolic BP, heart rate, respiratory rate, temperature, oxygen saturation
- **Clinical:** Chief complaint category (binned), arrival mode
- **Target:** ESI level (1-5, ordinal)
---
 
## Methodology
 
### Problem Statement
Current ED triage relies almost entirely on unaided human judgment, leading to inter-rater variability and documented undertriage of high-risk patients. We develop a machine learning model to predict acuity level from objectively measured vital signs and demographics, supporting (not replacing) clinician judgment.
 
### Approach
1. **Data Preprocessing:** Imputation of missing values, outlier handling, feature scaling
2. **Feature Engineering:** Vital sign ratios, age stratification, arrival mode encoding
3. **Model Selection:** XGBoost (gradient boosting) chosen for:
   - Non-linear relationships between vitals and acuity
   - Feature importance interpretability
   - Robustness to missing data
4. **Evaluation:** Stratified train/test split (80/20), cross-validation, per-class precision/recall
### Key Results
- **Overall Accuracy:** 74.3% on held-out test set
- **High-acuity sensitivity (ESI 1-2):** 81% (prioritizes patient safety)
- **Feature Importance:** Heart rate, systolic BP, respiratory rate are top predictors
- **Limitations:** Model trained on US data; generalization to other ED populations not validated
---
 
## Reproducibility
 
### Notebook Execution
The Kaggle notebook is fully self-contained and runs end-to-end without external dependencies beyond Python standard library + required packages (listed in notebook).
 
To reproduce locally:
1. Follow "Quick Start" above
2. Run `jupyter notebook triagegeist_notebook.ipynb`
3. All outputs (plots, metrics, model) regenerate
### Code Transparency
- All modeling steps are logged
- Hyperparameter choices are justified in comments
- Train/test split is seeded (`random_state=42`)
- No hardcoded paths; all data is fetched dynamically
---
 
## Clinical Context
 
**Why This Matters:**
- ~150 million ED visits annually in the U.S.
- Inter-rater reliability for ESI is 0.74–0.85 (substantial variability)
- Systematic undertriage of certain demographics documented in literature
- AI can flag overlooked risk and reduce cognitive load on overburdened clinicians
**Limitations & Caveats:**
- Model predictions are *support*, not replacement for clinical judgment
- Trained on US-based cohort; may not generalize to other healthcare systems
- Chief complaint is binned to broad categories; free-text analysis not included
- Model cannot capture unmeasured clinical judgment (e.g., "sick appearance")
---
 
## Directory Structure
 
```
triagegeist/
├── triagegeist_notebook.ipynb       # Main analysis (Kaggle)
├── train.py                         # Standalone training script
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── data/
│   └── fetch_nhamcs.py             # NHAMCS data fetching (optional)
└── models/
    └── xgboost_triage_v1.pkl       # Serialized trained model
```
 
---
 
## Requirements
 
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
seaborn==0.12.2
requests==2.31.0
```
 
Install with:
```bash
pip install -r requirements.txt
```
 
---
 
## License
 
This project is provided as-is for educational and research purposes in the context of the Triagegeist competition. All code is open-source under the MIT License.
 
---
 
## Citation
 
If you use this code or findings, please cite:
 
```bibtex
@competition{triagegeist2026,
  title = {Triagegeist: AI in Emergency Triage},
  author = {Feranmilux and Laitinen-Fredriksson Foundation},
  year = {2026},
  url = {https://kaggle.com/competitions/triagegeist}
}
```
 
---
 
## Contact
 
Questions or issues? Reach out via:
- GitHub: [@Feranmilux](https://github.com/Feranmilux)
- LinkedIn: [Toluwase Olufemi](https://linkedin.com/in/feranmiolufemi)
---
 
**Last Updated:** June 2026
