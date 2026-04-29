# ML Customer Prediction

> ⚠️ **Repo visibility note** — private until I confirm (1) data licensing
> permits public redistribution, and (2) any teammates from the class are OK
> with publication. The original `train.csv` and `test.csv` are intentionally
> excluded via `.gitignore` regardless.

A classification project predicting customer behavior using gradient boosting
and SHAP for explainability. Built for an MSBA machine learning class; the
final deliverable was a research poster.

## What's in It

- **EDA** — distribution of the target variable, age, and other key features
- **Preprocessing** — handling of missing values, encoding, train/test split
- **Modeling** — XGBoost classifier as the primary model
- **Explainability** — SHAP values to understand which features drive
  individual predictions and overall model behavior
- **Two notebooks**:
  - `Final_Code.ipynb` — full development notebook
  - `FinalPosterCodeWithLabels.ipynb` — cleaner, labeled version used to
    generate the final poster figures

## Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data handling
- `xgboost` — gradient boosting model
- `shap` — feature importance & explainability
- `scikit-learn` — preprocessing, metrics, train/test split
- `pregress` — regression utility (per the original install)
- `matplotlib`, `seaborn` — visualization

## Data

The training and test CSVs are not stored in this repository (they are
~65 MB and ~21 MB respectively — over GitHub's recommended size threshold).
Place `train.csv` and `test.csv` in the repository root before running.

## Run It

```bash
pip install pandas numpy xgboost shap scikit-learn matplotlib seaborn
jupyter notebook Final_Code.ipynb
```

## Files

| File | What it is |
| --- | --- |
| `Final_Code.ipynb` | Full EDA, preprocessing, modeling, evaluation |
| `FinalPosterCodeWithLabels.ipynb` | Cleaned-up version used for poster figures |
| `Customer Prediction Classification-2.png` | Final poster |

## Author

Nefeli Zafeiri — University of Miami Herbert Business School, MSBA.
