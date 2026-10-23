# Feature Engineering & Preprocessing — Handoff

This file is a short handoff for the Logistic Regression and SVM teammates.

The full project workflow is documented in:

```text
docs/TEAM_GUIDE.md
```

This file only covers what comes out of `02_feature_engineering_preprocessing.ipynb` and what the modelling notebooks should use next.

---

## Input to This Stage

The notebook starts from:

```text
data/processed/cleaned_shots.csv
```

Expected checks:

```text
12,374 shots
14 columns
0 missing values
0 duplicate shot IDs
```

Target distribution:

```text
No goal: 10,991  (88.82%)
Goal:     1,383  (11.18%)
```

---

## What This Stage Produces

The important outputs are:

```text
data/processed/modeling_dataset.csv
data/processed/model_split.csv
src/preprocessing.py
```

These three files are the shared handoff to both model notebooks.

---

## 1. `modeling_dataset.csv`

This is the feature-engineered dataset used for modelling.

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

The two feature settings are:

### Location-only

```python
[
    "distance",
    "angle"
]
```

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

`shot_statsbomb_xg` is included only for later benchmarking and must not be used as a training feature.

The file is intentionally saved before full scaling and one-hot encoding so preprocessing can be fitted correctly inside each model pipeline and each cross-validation fold.

---

## 2. `model_split.csv`

This file defines the fixed shared train/test assignment.

It contains:

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

The split is grouped by `match_id`, so all shots from the same match stay together.

Current split:

```text
Training shots:   9,900
Testing shots:    2,474

Training matches:   400
Testing matches:    100

Match overlap:         0

Training goal rate: 11.17%
Testing goal rate:  11.20%
```

Both model notebooks must use this file.

Do not create another train/test split.

---

## 3. `src/preprocessing.py`

This file contains the shared feature definitions and preprocessing used by both models.

Important shared objects include:

```python
LOCATION_FEATURES
CONTEXT_FEATURES
TARGET
NUMERICAL_FEATURES
CATEGORICAL_FEATURES
BOOLEAN_FEATURES
```

Important functions include:

```python
make_location_preprocessor()
make_context_preprocessor()
```

The preprocessing rules are:

```text
distance, angle
→ StandardScaler

shot_body_part, shot_technique, shot_type, play_pattern
→ OneHotEncoder(handle_unknown="ignore")

shot_first_time, under_pressure
→ Boolean to 0/1
```

Each helper returns a new, unfitted preprocessor.

---

## How the Model Notebooks Should Use This

Both `03_logistic_regression.ipynb` and `04_svm.ipynb` should:

1. load `modeling_dataset.csv`
2. load `model_split.csv`
3. recover the exact shared train/test observations
4. import preprocessing from `src/preprocessing.py`
5. build preprocessing inside the sklearn model pipeline
6. train both location-only and context-aware versions
7. save predictions/results for `05_model_evaluation.ipynb`

Example structure:

```python
from sklearn.pipeline import Pipeline
from src.preprocessing import make_context_preprocessor

pipeline = Pipeline([
    ("preprocessing", make_context_preprocessor()),
    ("model", ...)
])
```

Keeping preprocessing inside the pipeline prevents leakage during cross-validation because scaling and encoding are fitted only on each fold's training data.

---

## Expected Shapes

With the current shared split:

### Location-only

```text
Train: (9900, 2)
Test:  (2474, 2)
```

### Context-aware

```text
Train: (9900, 28)
Test:  (2474, 28)
```

The exact number of one-hot columns may vary slightly inside individual CV folds if a category is absent from one fold.

---

## Please Do Not Change These Independently

For a fair comparison between Logistic Regression and SVM:

- do not create a new split
- do not redefine the shared feature sets
- do not create separate preprocessing logic
- do not use `shot_statsbomb_xg` as a training feature
- do not fit scaling/encoding globally before cross-validation
- keep shot IDs in saved prediction files

---

## Handoff Summary

Use:

```text
data/processed/modeling_dataset.csv
data/processed/model_split.csv
src/preprocessing.py
```

Then train:

```text
Logistic Regression:
- location-only
- context-aware

SVM:
- location-only
- context-aware
```

Save the outputs needed by:

```text
notebooks/05_model_evaluation.ipynb
```
