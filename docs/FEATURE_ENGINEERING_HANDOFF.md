# Feature Engineering & Preprocessing — Handoff Guide

This guide explains the outputs of `02_feature_engineering_preprocessing.ipynb` and how the modelling notebooks should use them.

The purpose of this stage is to take the cleaned shot dataset, create the final modelling features, define one shared train/test split, and provide one shared preprocessing setup for both Logistic Regression and SVM.

---

## Input

This notebook starts from:

```text
data/processed/cleaned_shots.csv
```

That file is produced by `01_data_cleaning.ipynb`.

The cleaned dataset should contain:

```text
12,374 shots
14 columns
0 missing values
0 duplicate shot IDs
```

The target distribution should be:

```text
No goal (0): 10,991  → 88.82%
Goal (1):     1,383  → 11.18%
```

If these values are substantially different, check the cleaning notebook and input file before continuing.

---

## What This Notebook Does

`02_feature_engineering_preprocessing.ipynb` handles four main things:

1. creates the spatial shot features
2. defines the location-only and context-aware feature sets
3. creates the shared modelling dataset and match-level train/test split
4. defines and demonstrates the preprocessing used later by both models

The notebook does not repeat data cleaning and does not train any model.

---

## Shot Location Processing

StatsBomb stores shot location as:

```text
[x, y]
```

on a 120 × 80 coordinate system.

The attacking goal centre is:

```text
(120, 40)
```

and the goalposts are approximated as:

```text
(120, 36)
(120, 44)
```

The stored `location` field is converted into numeric `x` and `y` coordinates so that spatial features can be calculated.

---

## Distance to Goal

The `distance` feature is the Euclidean distance between the shot location and the centre of the attacking goal.

It is measured in **StatsBomb coordinate units**, not metres.

Distance is used because shots taken closer to goal generally have a higher scoring probability.

---

## Shot Angle

The `angle` feature represents the opening angle between the two goalposts from the shot location.

This complements distance because two shots can be equally far from goal while having very different shooting angles.

The model uses the angle in radians.

`angle_degrees` may also appear in the notebook for readability, but it is not used as an additional model feature because it contains the same information.

---

## Feature Sets

### Location-only

```python
[
    "distance",
    "angle"
]
```

This is the baseline feature set.

### Context-aware

```python
[
    "distance",
    "angle",
    "shot_body_part",
    "shot_technique",
    "shot_first_time",
    "shot_type",
    "play_pattern",
    "under_pressure"
]
```

This version adds contextual information so we can test whether it improves prediction beyond location alone.

### Target

```text
goal
```

The following columns are not model inputs:

```text
id
match_id
team
player
shot_outcome
shot_statsbomb_xg
```

`id` and `match_id` are kept for tracking and splitting.

`shot_outcome` is excluded because it directly determines the target.

`shot_statsbomb_xg` is kept only for later benchmarking against StatsBomb's xG estimate.

---

## Shared Modelling Dataset

The notebook exports:

```text
data/processed/modeling_dataset.csv
```

It contains:

```text
id
match_id
distance
angle
shot_body_part
shot_technique
shot_first_time
shot_type
play_pattern
under_pressure
goal
shot_statsbomb_xg
```

This file is intentionally saved **before scaling and one-hot encoding**.

The model notebooks should fit preprocessing only on training data, and during cross-validation the preprocessing should be refitted inside each training fold. Saving a globally transformed dataset would make that harder to do correctly and could introduce leakage.

---

## Shared Train/Test Split

The notebook also exports:

```text
data/processed/model_split.csv
```

The split is based on `match_id`, so every match belongs entirely to either training or testing.

This avoids having shots from the same match appear in both sets.

The split is created using:

```text
StratifiedGroupKFold
5 folds
shuffle = True
random_state = 42
```

The first fold is used as the fixed test set.

The current split is:

```text
Training shots:   9,900
Testing shots:    2,474

Training matches:   400
Testing matches:    100

Match overlap:         0
```

The target balance remains very similar:

```text
Training goal rate: 11.17%
Testing goal rate:  11.20%
```

The split is approximately 80/20 rather than exactly 80/20 by shot count because complete matches must stay together and matches contain different numbers of shots.

`model_split.csv` contains:

```text
id
match_id
split
```

where `split` is either:

```text
train
```

or:

```text
test
```

The Logistic Regression and SVM notebooks should both use this file instead of creating their own train/test split.

---

## Shared Preprocessing

The reusable preprocessing logic is stored in:

```text
src/preprocessing.py
```

Both model notebooks should import from this file so they use the same transformations.

### Numerical features

```text
distance
angle
```

These are standardized with:

```text
StandardScaler
```

because they are on different numerical scales and both SVM and Logistic Regression can be sensitive to feature scale.

### Categorical features

```text
shot_body_part
shot_technique
shot_type
play_pattern
```

These are transformed with:

```text
OneHotEncoder(handle_unknown="ignore")
```

One-hot encoding is used because these categories do not have a meaningful numerical order.

`handle_unknown="ignore"` prevents errors if a category appears in a validation or test subset that was not present when the encoder was fitted.

### Boolean features

```text
shot_first_time
under_pressure
```

These are converted to:

```text
False → 0
True  → 1
```

---

## `preprocessing.py`

The shared module contains the common feature definitions and reusable preprocessing functions.

Important objects include:

```python
LOCATION_FEATURES
CONTEXT_FEATURES
TARGET
NUMERICAL_FEATURES
CATEGORICAL_FEATURES
BOOLEAN_FEATURES
```

and:

```python
make_location_preprocessor()
make_context_preprocessor()
```

Each function returns a new, unfitted preprocessor.

The model notebooks should create preprocessing through these functions instead of rewriting their own scaler/encoder logic.

---

## Using Preprocessing in the Model Notebooks

Preprocessing should stay inside the sklearn model pipeline.

For example:

```python
from sklearn.pipeline import Pipeline
from src.preprocessing import make_context_preprocessor

pipeline = Pipeline([
    ("preprocessing", make_context_preprocessor()),
    ("model", ...)
])
```

This is especially important during cross-validation.

When preprocessing is part of the pipeline, each fold learns scaling and encoding only from that fold's training data. This prevents information from the validation fold leaking into training.

---

## Expected Processed Shapes

With the current fixed split:

### Location-only

```text
Training: (9900, 2)
Testing:  (2474, 2)
```

### Context-aware

```text
Training: (9900, 28)
Testing:  (2474, 28)
```

The context-aware feature count comes from the categories present in the training data, so the exact one-hot column count may differ slightly inside individual cross-validation folds.

---

## What Logistic Regression Should Use

`03_logistic_regression.ipynb` should use:

```text
data/processed/modeling_dataset.csv
data/processed/model_split.csv
src/preprocessing.py
```

It should train both:

```text
location-only Logistic Regression
context-aware Logistic Regression
```

It should not:

- repeat cleaning
- recreate distance or angle differently
- generate another train/test split
- define different preprocessing
- use `shot_statsbomb_xg` as a predictor

The notebook should save its trained pipelines and the predictions/results needed later by `05_model_evaluation.ipynb`.

---

## What SVM Should Use

`04_svm.ipynb` should use the same:

```text
data/processed/modeling_dataset.csv
data/processed/model_split.csv
src/preprocessing.py
```

It should train both:

```text
location-only SVM
context-aware SVM
```

The SVM notebook should use the exact same train/test observations and preprocessing definitions as Logistic Regression so the model comparison remains fair.

---

## Outputs From This Stage

The important outputs of this stage are:

```text
data/processed/modeling_dataset.csv
data/processed/model_split.csv
src/preprocessing.py
```

These three files are the handoff from feature engineering/preprocessing to model training.

In short:

```text
cleaned_shots.csv
        ↓
feature engineering
        ↓
modeling_dataset.csv
        +
model_split.csv
        +
preprocessing.py
        ↓
Logistic Regression / SVM
```

---

## Quick Consistency Check

Before starting model training, the following values should match:

```text
Total shots:            12,374
Total matches:             500

Training shots:           9,900
Testing shots:            2,474

Training matches:           400
Testing matches:            100

Match overlap:                 0

Training goal rate:        11.17%
Testing goal rate:         11.20%
```

If these are different, check the input dataset, `model_split.csv`, and preprocessing setup before continuing.

---

## Final Note

The most important requirement for the next modelling stage is consistency.

Both Logistic Regression and SVM should use:

```text
the same engineered data
the same train/test split
the same preprocessing rules
```

That way, any difference in final performance reflects the models and feature settings rather than differences in the experimental setup.
