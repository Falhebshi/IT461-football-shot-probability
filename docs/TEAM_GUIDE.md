# IT461 Football Shot Probability — Team Guide

This guide explains how the project is organized, how data moves through the pipeline, and how the notebooks and shared files are meant to work together.

The main principle is simple:

**Logistic Regression and SVM should use the same data, the same split, the same preprocessing, and the same evaluation setup so their results can be compared fairly.**

---

## Project Overview

The project predicts whether a **non-penalty football shot results in a goal**.

The target is binary:

- `goal = 1` if the shot results in a goal
- `goal = 0` otherwise

The predicted probability of the goal class is interpreted as the model's **Expected Goals (xG)** estimate.

We compare two feature settings:

### Location-only
- `distance`
- `angle`

### Context-aware
- `distance`
- `angle`
- `shot_body_part`
- `shot_technique`
- `shot_first_time`
- `shot_type`
- `play_pattern`
- `under_pressure`

The two models are:

- Logistic Regression
- Support Vector Machine (SVM)

`shot_statsbomb_xg` is kept only for later benchmarking. It must not be used as a training feature because it is already an existing xG estimate.

---

## Project Structure

```text
IT461-football-shot-probability/
│
├── data/
│   ├── metadata/
│   │   ├── statsbomb_dataset_summary.csv
│   │   └── statsbomb_selected_matches.csv
│   │
│   ├── raw/
│   │   └── statsbomb_non_penalty_shots_raw.csv
│   │
│   └── processed/
│       ├── cleaned_shots.csv
│       ├── modeling_dataset.csv
│       └── model_split.csv
│
├── models/
│
├── notebooks/
│   ├── 00_data_preparation.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_svm.ipynb
│   └── 05_model_evaluation.ipynb
│
├── results/
│
├── src/
│   └── preprocessing.py
│
├── docs/
│   ├── TEAM_GUIDE.md
│   └── FEATURE_ENGINEERING_HANDOFF.md
│
├── .gitignore
└── README.md
```

The notebooks are separated by task so each stage has a clear responsibility and teammates can work independently without duplicating the same logic.

---

## Overall Workflow

```text
Raw StatsBomb data
        ↓
00_data_preparation.ipynb
        ↓
statsbomb_non_penalty_shots_raw.csv
        ↓
01_data_cleaning.ipynb
        ↓
cleaned_shots.csv
        ↓
02_feature_engineering_preprocessing.ipynb
        ↓
modeling_dataset.csv
model_split.csv
src/preprocessing.py
        ↓
03_logistic_regression.ipynb
04_svm.ipynb
        ↓
saved models + results
        ↓
05_model_evaluation.ipynb
```

Each notebook should use the official output of the previous stage instead of repeating earlier work. This keeps the pipeline consistent across the team.

---

## 00 — Data Preparation

`00_data_preparation.ipynb` prepares the original StatsBomb shot data and produces:

```text
data/raw/statsbomb_non_penalty_shots_raw.csv
```

The current dataset contains:

```text
12,374 non-penalty shots
500 matches
```

Keeping this stage separate gives the rest of the project one stable raw input.

---

## 01 — Data Cleaning

`01_data_cleaning.ipynb` handles the basic cleaning before feature engineering.

Its main tasks are:

- inspect missing values
- fill missing Boolean flags
- create the binary target
- check duplicates
- verify the class distribution
- export the cleaned dataset

The missing Boolean values are:

```text
shot_first_time    8,494 missing
under_pressure     9,278 missing
```

These are filled with `False`.

The target is created from `shot_outcome`:

```text
Goal          → 1
Anything else → 0
```

The final class distribution is:

```text
No goal: 10,991  (88.82%)
Goal:     1,383  (11.18%)
```

The cleaned dataset is exported to:

```text
data/processed/cleaned_shots.csv
```

This becomes the input to the next stage.

---

## 02 — Feature Engineering and Preprocessing

`02_feature_engineering_preprocessing.ipynb` starts from `cleaned_shots.csv`.

Its job is to:

- create the spatial features `distance` and `angle`
- define the location-only and context-aware feature sets
- export the shared modelling dataset
- create the shared match-level train/test split
- define and demonstrate the preprocessing used by both models

The notebook does not train any model.

### Spatial Features

StatsBomb stores shot locations as `[x, y]` coordinates on a 120 × 80 pitch.

The attacking goal centre is:

```text
(120, 40)
```

and the goalposts are approximated as:

```text
(120, 36)
(120, 44)
```

`distance` is the Euclidean distance from the shot to the goal centre.

`angle` is the opening angle between the two goalposts from the shot location.

Using both gives a stronger description of shot location than distance alone, because two shots can be equally far from goal while having very different shooting angles.

### Shared Modelling Dataset

The notebook exports:

```text
data/processed/modeling_dataset.csv
```

This file contains the engineered features, target, stable IDs, and `shot_statsbomb_xg` for later benchmarking.

It is intentionally saved before full scaling and one-hot encoding. Scaling and encoding should be fitted only on training data and should be refitted inside each cross-validation fold.

### Shared Train/Test Split

The notebook also exports:

```text
data/processed/model_split.csv
```

The split is grouped by `match_id`, so all shots from the same match stay together.

This avoids placing related shots from one match in both training and testing.

The current split is:

```text
Training shots:   9,900
Testing shots:    2,474

Training matches:   400
Testing matches:    100

Match overlap:         0

Training goal rate: 11.17%
Testing goal rate:  11.20%
```

The split is approximately 80/20 rather than exactly 80/20 by shot count because complete matches must stay together.

More details for the modelling teammates are available in:

```text
docs/FEATURE_ENGINEERING_HANDOFF.md
```

---

## Shared Preprocessing

Reusable preprocessing is stored in:

```text
src/preprocessing.py
```

This ensures both Logistic Regression and SVM use the same feature definitions and transformations.

### Numerical features

```text
distance
angle
```

These are standardized with `StandardScaler`.

### Categorical features

```text
shot_body_part
shot_technique
shot_type
play_pattern
```

These are one-hot encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

### Boolean features

```text
shot_first_time
under_pressure
```

These are converted from Boolean values to numeric 0/1 values.

The module provides reusable functions such as:

```python
make_location_preprocessor()
make_context_preprocessor()
```

These return new, unfitted preprocessors so they can be safely used inside model pipelines.

---

## 03 — Logistic Regression

`03_logistic_regression.ipynb` should:

1. load `modeling_dataset.csv`
2. load `model_split.csv`
3. recreate the fixed train/test sets
4. import shared preprocessing from `src/preprocessing.py`
5. train a location-only Logistic Regression model
6. train a context-aware Logistic Regression model
7. perform tuning/validation
8. evaluate both on the shared test set
9. save trained pipelines
10. save predictions/results needed for final evaluation

The Logistic Regression notebook should not repeat cleaning, feature engineering, splitting, or define separate preprocessing.

---

## 04 — SVM

`04_svm.ipynb` should follow the same structure:

1. load `modeling_dataset.csv`
2. load `model_split.csv`
3. recreate the same train/test sets
4. import the same shared preprocessing
5. train a location-only SVM
6. train a context-aware SVM
7. perform tuning/validation
8. evaluate on the same shared test set
9. save trained pipelines
10. save predictions/results needed for final evaluation

Using the same data and preprocessing makes the model comparison meaningful.

---

## Model Pipelines

Preprocessing should stay inside the sklearn model pipeline.

Example:

```python
Pipeline([
    ("preprocessing", make_context_preprocessor()),
    ("model", ...)
])
```

This is important during cross-validation because the scaler and encoder must learn only from each fold's training data.

Fitting preprocessing before cross-validation could leak information from validation data into training.

---

## Models and Results

Trained model pipelines should be saved in:

```text
models/
```

For example:

```text
logistic_location.joblib
logistic_context.joblib
svm_location.joblib
svm_context.joblib
```

Saved predictions and metrics should go in:

```text
results/
```

Prediction files should preserve the unique shot `id` so outputs from different models can be aligned during final evaluation.

---

## 05 — Model Evaluation

`05_model_evaluation.ipynb` is the final required comparison stage.

It should compare:

- location-only Logistic Regression
- context-aware Logistic Regression
- location-only SVM
- context-aware SVM

The notebook should evaluate all models on the same fixed test set.

Because the target is imbalanced, accuracy alone is not enough. The evaluation should include:

- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- ROC curve
- precision-recall curve

Probability-based evaluation should also be included where appropriate because the main xG output is a probability.

The final notebook should also compare predicted xG values with `shot_statsbomb_xg` where useful, select the best model, and summarize the main findings.

---

## Shared Rules

To keep the project consistent:

- use `modeling_dataset.csv` as the shared modelling data
- use `model_split.csv` for the fixed train/test assignment
- use `src/preprocessing.py` for shared preprocessing
- do not create a new split in individual model notebooks
- do not use `shot_statsbomb_xg` as a training feature
- keep preprocessing inside the sklearn pipeline during cross-validation
- preserve shot IDs in saved predictions
- save the outputs needed by `05_model_evaluation.ipynb`

---

## Running Order

To reproduce the full project:

```text
1. 00_data_preparation.ipynb
2. 01_data_cleaning.ipynb
3. 02_feature_engineering_preprocessing.ipynb
4. 03_logistic_regression.ipynb
5. 04_svm.ipynb
6. 05_model_evaluation.ipynb
```

---

## Useful Consistency Checks

Current dataset:

```text
Total shots:       12,374
Total matches:        500

No-goal shots:     10,991
Goal shots:         1,383

No-goal:            88.82%
Goal:               11.18%
```

Current shared split:

```text
Training shots:      9,900
Testing shots:       2,474

Training matches:      400
Testing matches:       100

Match overlap:           0

Training goal rate:  11.17%
Testing goal rate:   11.20%
```

Current processed dimensions:

```text
Location train: (9900, 2)
Location test:  (2474, 2)

Context train:  (9900, 28)
Context test:   (2474, 28)
```

If someone gets substantially different values, check the dataset version, file paths, split file, and preprocessing setup before continuing.

---

## In Short

```text
modeling_dataset.csv
        +
model_split.csv
        +
preprocessing.py
        ↓
Logistic Regression + SVM
        ↓
05_model_evaluation.ipynb
```

**Same data + same split + same preprocessing + same evaluation setup = fair model comparison.**
