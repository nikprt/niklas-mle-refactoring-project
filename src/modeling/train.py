# build_model_pipeline()
# run_grid_search()

from __future__ import annotations 

from dataclasses import dataclass 

import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import (
    ID_COLUMN, 
    PARAM_GRID, 
    RANDOM_STATE, 
    TEST_SIZE, 
    TARGET_COLUMN,
    LEAKAGE_COLUMNS
)

@dataclass
class TrainingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_ids: pd.DataFrame # 'id' column of the test set
  
  
def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    leakage_columns: list[str] = LEAKAGE_COLUMNS,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features and target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        leakage_columns (list[str]): List of columns to drop due to data leakage.
        id_column (str): The name of the ID column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
    """
    
    cols_to_drop = set(leakage_columns | {target_column, id_column})
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    return df[feature_cols], df[target_column]


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainingData:
    
    """
    Split the DataFrame into training and testing sets.
    """
    
    X, y = split_features_and_target(df)
    ids = df[[ID_COLUMN]]

    X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state
    )
    return TrainingData(X_train, X_test, y_train, y_test, test_ids)

def build_model_pipeline() -> Pipeline:
    """
    Build a machine learning pipeline for regression.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    
    pipeline = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "model", 
            TransformedTargetRegressor(
                regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                transformer=StandardScaler(),
            ),
        )
    ])
    
    return pipeline

def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = PARAM_GRID,
    cv: int = 5,
) -> GridSearchCV:
    """
    Run a grid search to find the best hyperparameters for the model.
    """
    
    grid_search = GridSearchCV(
        build_model_pipeline(),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def train(df: pd.DataFrame) -> tuple[Pipeline, TrainingData, GridSearchCV]:
    """
    Train the model using the provided DataFrame.
    """
    data = make_train_test_split(df)
    grid_search = run_grid_search(data.X_train, data.y_train)
    best_model = grid_search.best_estimator_  # GridSearchCV already refits on all of X_train
    return best_model, data, grid_search
