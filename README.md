# Football Expected Goals (xG) Prediction

IT461 Practical Machine Learning | Group 10 (Marma)
Haya Alomar · Sara Aljuraybah · Deema Alquwaei · Fatimah Alhebshi · Aliyah Alharbi

## Overview

This project estimates the probability that a non-penalty football shot results in a goal. This probability is the shot's **expected goals (xG)** value: an xG of 0.20 means that similar shots are scored about 20% of the time.

### Research question

**How accurately can machine learning models estimate the probability that a shot results in a goal, and how much do contextual features improve prediction beyond shot location?**

- **Task:** supervised binary classification (`goal = 1`, otherwise `0`)
- **Input:** shot location (distance and angle) and, in the context-aware setting, body part, technique, first-time, shot type, play pattern and pressure
- **Output:** the predicted probability of a goal, i.e. the shot's xG

## Dataset

12,374 non-penalty shots from 500 matches of **StatsBomb Open Data**, extracted with `statsbombpy` (seed 42). Only 11.2% of the shots are goals. StatsBomb's own xG is kept as a benchmark only and is never used for training.

## How to run

Open **`notebooks/IT461_Final_Implementation.ipynb`** in Google Colab and choose *Runtime → Run all* (about 10 minutes). The notebook downloads the data from this repository, installs scikit-learn 1.6.1, and runs the full pipeline: cleaning, feature engineering, training, tuning, evaluation and model selection. It saves its outputs to `results/` and `models/`.

## Approach

- **Features:** location-only (distance, angle) vs context-aware (8 features)
- **Split:** by match, 400 training matches (9,900 shots) and 100 test matches (2,474 shots), with the same goal rate in both
- **Models:** Logistic Regression and SVM, each tuned with grid search on 3 match-grouped cross-validation folds
- **Model selection:** lowest out-of-fold log loss on the training data, using the one-standard-error rule

## Main results (test set)

| Model | Log loss | ROC AUC | PR AUC |
|---|---|---|---|
| LR location-only | 0.306 | 0.754 | 0.337 |
| **LR context-aware (selected)** | **0.284** | **0.797** | **0.431** |
| SVM location-only | 0.307 | 0.753 | 0.337 |
| SVM context-aware | 0.285 | 0.797 | 0.427 |
| StatsBomb xG (benchmark) | 0.265 | 0.830 | 0.512 |

- Context features clearly improve both models, and the improvement is not chance (match-level bootstrap).
- Logistic Regression and SVM perform almost the same; the tuned SVM chose a linear kernel.
- Location alone overrates headers and shots under pressure and underrates counter-attacks; context corrects this.

## Repository structure

| Folder | Contents |
|---|---|
| `notebooks/` | `IT461_Final_Implementation.ipynb` (**the final notebook**) and the earlier per-stage notebooks (00–04) |
| `data/raw/` | Raw non-penalty shots extracted from StatsBomb |
| `data/processed/` | Cleaned data, modelling dataset and the fixed train/test split (`model_split.csv`) |
| `data/metadata/` | Selected matches and a dataset summary |
| `results/` | Test predictions, metrics and comparison tables produced by the final notebook |
| `models/` | The four fitted models (`.joblib`) |
| `src/` | Shared preprocessing code used by the per-stage notebooks |
| `docs/` | Team guide and feature-engineering notes |

The per-stage notebooks (00–04) show how the work was developed. The final notebook is self-contained, and its results are the official ones.
