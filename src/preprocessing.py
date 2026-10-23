"""Shared preprocessing definitions for the xG models.

This file keeps the Logistic Regression and SVM notebooks consistent by
using the same feature sets and the same preprocessing transformations.
The preprocessors returned here are unfitted and must be fitted using
training data only.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


# -----------------------------------------------------------------------------
# Shared feature definitions
# -----------------------------------------------------------------------------

# Features used by the location-only model
LOCATION_FEATURES = [
    "distance",
    "angle",
]

# Features used by the context-aware model
CONTEXT_FEATURES = [
    "distance",
    "angle",
    "shot_body_part",
    "shot_technique",
    "shot_first_time",
    "shot_type",
    "play_pattern",
    "under_pressure",
]

# Binary target: 1 = goal, 0 = no goal
TARGET = "goal"

# StatsBomb xG is not a training feature; it is kept only for later comparison
BENCHMARK = "shot_statsbomb_xg"


# -----------------------------------------------------------------------------
# Feature groups used by the context-aware preprocessor
# -----------------------------------------------------------------------------

# Numerical features are standardized because they are measured on
# different numerical scales and both LR and SVM are scale-sensitive.
NUMERICAL_FEATURES = [
    "distance",
    "angle",
]

# Categorical features are one-hot encoded so categories are represented
# without introducing an artificial numerical order.
CATEGORICAL_FEATURES = [
    "shot_body_part",
    "shot_technique",
    "shot_type",
    "play_pattern",
]

# Boolean features are converted from True/False to 1/0.
BOOLEAN_FEATURES = [
    "shot_first_time",
    "under_pressure",
]


def boolean_to_int(x):
    """Convert Boolean feature values from True/False to 1/0."""
    return x.astype(int)


# -----------------------------------------------------------------------------
# Reusable preprocessing functions
# -----------------------------------------------------------------------------

def make_location_preprocessor():
    """Create an unfitted preprocessor for the location-only feature set."""

    # Distance and angle are the only inputs, so both are standardized.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )


def make_context_preprocessor():
    """Create an unfitted preprocessor for the context-aware feature set."""

    # Convert the two Boolean columns to binary numerical values.
    boolean_transformer = FunctionTransformer(
        boolean_to_int,
        feature_names_out="one-to-one"
    )

    # Apply the appropriate transformation to each feature type.
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "bool",
                boolean_transformer,
                BOOLEAN_FEATURES,
            ),
        ]
    )
