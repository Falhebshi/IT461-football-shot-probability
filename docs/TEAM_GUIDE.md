# IT461 Football Shot Probability — Team Guide

This guide explains how the project is organized, how data moves through the pipeline, and how the different notebooks and shared files are meant to work together.

The main goal of the structure is simple: **Logistic Regression and SVM should use the same data, the same split, and the same preprocessing**, so that their results can be compared fairly.

---

## Project Overview

The project predicts whether a **non-penalty football shot results in a goal**.

The target is binary:

- `goal = 1` if the shot results in a goal
- `goal = 0` otherwise

The predicted probability of the goal class is interpreted as the model's **Expected Goals (xG)** estimate.

We evaluate two feature settings:

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

`shot_statsbomb_xg` is kept only for later comparison with StatsBomb's own xG values. It must **not** be used as a training feature.

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
├── .gitignore
└── README.md
```

The notebooks are separated by task so each stage has a clear responsibility and teammates can work independently without duplicating each other's code.

The general flow is:

```text
Raw StatsBomb data
        ↓
00_data_preparation.ipynb
        ↓
raw shot CSV
        ↓
01_data_cleaning.ipynb
        ↓
cleaned_shots.csv
        ↓
02_feature_engineering_preprocessing.ipynb
        ↓
modeling_dataset.csv
model_split.csv
        ↓
03_logistic_regression.ipynb
04_svm.ipynb
        ↓
saved models + results
        ↓
05_model_evaluation.ipynb
```

---

## 00 — Data Preparation

`00_data_preparation.ipynb` is responsible for preparing the original StatsBomb data and producing the raw non-penalty shot dataset:

```text
data/raw/statsbomb_non_penalty_shots_raw.csv
```

The current dataset contains:

```text
12,374 shots
500 matches
```

Keeping this stage separate gives the rest of the project one stable raw input instead of requiring every teammate to repeat the StatsBomb extraction process.

---

## 01 — Data Cleaning

`01_data_cleaning.ipynb` loads the raw shot dataset and handles the basic cleaning needed before feature engineering.

The main cleaning steps are:

- inspect missing values
- fill missing Boolean flags
- create the binary target
- check duplicates
- verify class distribution
- export the cleaned dataset

The two columns with missing values are:

```text
shot_first_time    8,494 missing
under_pressure     9,278 missing
```

These are treated as Boolean event flags, so missing values are replaced with `False`.

The target is created from `shot_outcome`:

```text
Goal          → 1
Anything else → 0
```

This gives:

```text
No goal: 10,991  (88.82%)
Goal:     1,383  (11.18%)
```

The final cleaned dataset has:

```text
12,374 rows
14 columns
0 missing values
0 duplicate shot IDs
```

The notebook exports:

```text
data/processed/cleaned_shots.csv
```

The next notebook starts from this file instead of repeating the cleaning steps.

---

## 02 — Feature Engineering and Preprocessing

`02_feature_engineering_preprocessing.ipynb` starts from:

```text
data/processed/cleaned_shots.csv
```

Its job is to create the actual model features, define the shared train/test split, and prepare the preprocessing logic used by both models.

### Shot Coordinates

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

The stored `location` value is converted into separate numeric `x` and `y` coordinates before calculating spatial features.

---

### Distance to Goal

`distance` is the Euclidean distance from the shot location to the centre of the attacking goal.

It is measured in **StatsBomb coordinate units**, not metres.

Distance is included because shot location strongly affects scoring probability: shots taken closer to goal are generally more dangerous.

---

### Shot Angle

`angle` represents the opening angle between the two goalposts from the location of the shot.

This adds information that distance alone cannot capture. Two shots can be equally far from goal, but a central shot can have a much wider view of the goal than a shot from a wide position.

The model uses the angle in radians.

`angle_degrees` may be created for readability, but it is not used as another model feature because it contains the same information as `angle`.

---

## Feature Sets

The location-only feature set is:

```python
[
    "distance",
    "angle"
]
```

This acts as the baseline.

The context-aware feature set is:

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

The purpose of having both settings is to test whether contextual information improves prediction beyond location alone.

The following columns are not used as model inputs:

```text
id
match_id
team
player
shot_outcome
shot_statsbomb_xg
```

`id` and `match_id` are kept only for tracking and splitting.

`shot_outcome` is excluded because it directly determines the target.

`shot_statsbomb_xg` is excluded because it is already an xG estimate and is only used later as a benchmark.

---

## Shared Modelling Dataset

After feature engineering, the notebook exports:

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

This file is intentionally saved **before full scaling and one-hot encoding**.

The reason is that scalers and encoders should be fitted only on training data. During cross-validation, they also need to be fitted separately inside each training fold. Saving a globally transformed dataset could introduce leakage.

---

## Shared Train/Test Split

The project uses a shared split so Logistic Regression and SVM are always trained and evaluated on the exact same observations.

Instead of splitting individual shots randomly, the split is grouped by:

```text
match_id
```

This means all shots from one match stay together:

```text
Match A → train
Match B → train
Match C → test
```

We avoid putting some shots from the same match in training and others in testing because shots from one match can share match-specific context.

The split is created using `StratifiedGroupKFold` with:

```text
5 folds
shuffle = True
random_state = 42
```

The first fold is used as the fixed test set.

This keeps complete matches together while also keeping the goal proportion very similar between training and testing.

The final split is:

```text
Training shots:   9,900
Testing shots:    2,474

Training matches:   400
Testing matches:    100

Match overlap:         0
```

Goal rates are:

```text
Training: 11.17%
Testing:  11.20%
```

The old shot-level split had 9,899 / 2,475 shots. The one-shot difference is expected because whole matches now have to stay together, so the split is approximately rather than perfectly 80/20 by shot count.

The assignment is saved to:

```text
data/processed/model_split.csv
```

with:

```text
id
match_id
split
```

Every model notebook should use this file instead of creating a new random split.

---

## Shared Preprocessing

The shared preprocessing logic is stored in:

```text
src/preprocessing.py
```

This keeps Logistic Regression and SVM consistent.

### Numerical Features

```text
distance
angle
```

These are standardized using `StandardScaler`.

This is useful because the two features are on very different numerical scales, and both SVM and Logistic Regression can be affected by feature scale.

### Categorical Features

```text
shot_body_part
shot_technique
shot_type
play_pattern
```

These are transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

One-hot encoding is used because the categories do not have a meaningful numerical order.

`handle_unknown="ignore"` prevents errors if a validation or test subset contains a category that was not present when the encoder was fitted.

### Boolean Features

```text
shot_first_time
under_pressure
```

These are converted to:

```text
False → 0
True  → 1
```

The module provides reusable functions such as:

```python
make_location_preprocessor()
make_context_preprocessor()
```

Each function returns a new, unfitted preprocessor.

This is important because preprocessing should be fitted inside each model pipeline rather than fitted once globally.

---

## Why Preprocessing Stays Inside the Model Pipeline

The model notebooks should use preprocessing like this:

```python
Pipeline([
    ("preprocessing", make_context_preprocessor()),
    ("model", ...)
])
```

This matters especially during cross-validation and hyperparameter tuning.

For every CV fold, the scaler and encoder should learn only from that fold's training data. If preprocessing were fitted before cross-validation, information from the validation fold could leak into the training process.

---

## 03 — Logistic Regression

`03_logistic_regression.ipynb` should:

1. load `modeling_dataset.csv`
2. load `model_split.csv`
3. recreate the fixed shared train/test sets
4. import preprocessing from `src/preprocessing.py`
5. train the location-only Logistic Regression model
6. train the context-aware Logistic Regression model
7. perform tuning/validation
8. evaluate both models on the shared test set
9. save the trained model pipelines
10. save predictions and metrics needed for final evaluation

The Logistic Regression notebook should not redo cleaning, feature engineering, splitting, or create its own preprocessing rules.

---

## 04 — SVM

`04_svm.ipynb` follows the same structure:

1. load `modeling_dataset.csv`
2. load `model_split.csv`
3. recreate the same shared train/test sets
4. import the same preprocessing
5. train the location-only SVM
6. train the context-aware SVM
7. perform tuning/validation
8. evaluate on the same shared test set
9. save the trained model pipelines
10. save predictions and metrics needed for final evaluation

Using the same dataset, split, and preprocessing means any difference in performance is much more likely to come from the modelling method rather than inconsistent experimental setup.

---

## Models and Results

Trained model pipelines should be saved inside:

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

Saving the full pipeline is preferable because it keeps preprocessing and the classifier together.

Model outputs used later for comparison should go inside:

```text
results/
```

For example:

```text
logistic_predictions.csv
svm_predictions.csv
logistic_metrics.csv
svm_metrics.csv
```

Prediction files should keep the unique shot `id` so results from different models can be aligned to the same observations.

---

## 05 — Model Evaluation

`05_model_evaluation.ipynb` is the final comparison stage.

It should load the saved results from Logistic Regression and SVM and compare all model/feature combinations on the same shared test set.

The final comparison should include:

- location-only Logistic Regression
- context-aware Logistic Regression
- location-only SVM
- context-aware SVM

The evaluation should compare:

- location-only vs context-aware within each model
- Logistic Regression vs SVM
- our predicted probabilities vs StatsBomb xG where appropriate
- the final best model

Because goals make up only about 11% of the dataset, accuracy alone is not enough.

The evaluation should include:

- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- ROC curve
- precision-recall curve

Probability-based evaluation should also be considered because the main output of an xG model is a probability rather than only a final 0/1 prediction.

The final notebook should end with a clear best-model selection and discussion of the main findings.

---

## Important Shared Rules

To keep the experiment consistent across the team:

- use `modeling_dataset.csv` as the shared modelling data
- use `model_split.csv` for the fixed train/test assignment
- use `src/preprocessing.py` for shared feature definitions and transformations
- do not create a new train/test split in individual model notebooks
- do not use `shot_statsbomb_xg` as a training feature
- keep preprocessing inside the sklearn model pipeline during cross-validation
- preserve shot IDs when saving predictions
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

Each notebook should produce the files required by the next stage.

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

The shared modelling setup is:

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

The main principle is:

**same data + same split + same preprocessing + same evaluation setup = fair model comparison**
