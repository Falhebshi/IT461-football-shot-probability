# Football Expected Goals (xG) Prediction

IT461 Practical Machine Learning Project

## Overview

This project investigates the use of machine learning to estimate the probability that a football shot results in a goal.

The predicted probability represents the shot's **Expected Goals (xG)** value. For example, an xG value of 0.20 indicates an estimated 20% probability that the shot will result in a goal.

### Research Question

**How accurately can machine learning models estimate the probability that a football shot results in a goal, and how much do contextual features improve prediction beyond shot location?**

The project is formulated as a **supervised binary classification problem**:

- **Input:** characteristics of a football shot, including its location and contextual information.
- **Output:** whether the shot resulted in a goal (`1`) or not (`0`).
- **Model output:** probability of scoring, interpreted as the shot's xG value.

## Dataset

The project uses publicly available football event data from **StatsBomb Open Data**.

The working dataset contains shots extracted from 500 selected matches. Penalty shots are excluded because penalties represent a substantially different scoring situation from ordinary shots.

Each row represents one non-penalty shot.

The repository keeps the original selected shot data separate from data generated during preprocessing and feature engineering.

### Data Files

`data/raw/statsbomb_non_penalty_shots_raw.csv`  
Raw non-penalty shot records used for the project.

`data/metadata/statsbomb_selected_matches.csv`  
Information about the StatsBomb matches included in the dataset.

`data/metadata/statsbomb_dataset_summary.csv`  
Summary statistics and information about the extracted dataset.

## Planned Features

The project will examine two main groups of information.

### Location Information

Shot location will be used to derive spatial features such as:

- Distance to goal
- Shooting angle

### Contextual Information

Contextual variables may include:

- Body part
- Shot technique
- Shot type
- Play pattern
- Whether the player was under pressure
- Whether the shot was taken first-time

Player, team, event, and match identifiers may be retained for analysis and traceability but are not intended to be initial model predictors.

StatsBomb's existing xG value will not be used as a training feature.
