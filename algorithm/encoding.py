import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# TODO: Smoothing
def target_encode(
    c: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    label: str = "label",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    is_straitified: bool = False,
) -> None:
    """Target Encoding function (inplace)

    Args:
        c (str): Name of category column
        train_df (pd.DataFrame): Train DataFrame
        test_df (pd.DataFrame, optional): Test DataFrame. Defaults to None.
        label (string, optional): Name of label column. Defaults to 'label'.
        n_splits (int, optional): The number of folds. Defaults to 5.
        shuffle (bool, optional): enable shuffle. Defaults to True.
        random_state (int, optional): seed number. Defaults to 42.
        is_straitified (bool,optional): valid straitified KFold. Defaults to True.

    """

    # Encoding test data with all the train data
    if test_df is not None:
        target_mean = train_df[[c, label]].groupby(c)[label].mean()
        test_df[f"target_{c}"] = test_df[c].map(target_mean)

    # Encoding train data

    ts = pd.Series(np.empty(train_df.shape[0]), index=train_df.index)

    if is_straitified:
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        for main_idx, rest_idx in folds.split(train_df, train_df[label]):
            target_mean = train_df[[c, label]].iloc[main_idx].groupby(c)[label].mean()
            ts[rest_idx] = train_df[c].iloc[rest_idx].map(target_mean)
    else:
        folds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for main_idx, rest_idx in folds.split(train_df):
            target_mean = train_df[[c, label]].iloc[main_idx].groupby(c)[label].mean()
            ts[rest_idx] = train_df[c].iloc[rest_idx].map(target_mean)

    train_df[f"target_{c}"] = ts
