import itertools
import pandas as pd

from fastcore.foundation import L
from sklearn.model_selection import train_test_split

# Typing
from typing import List, Union
from pandas import DataFrame


def balanced_train_test_split(
    frame: DataFrame,
    by: str = "label",
    test_perc: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
):
    """Split frame into train and test. Test is balanced by taking test_perc of smallest group as size."""
    by = by.replace(" ", "").split(",")

    pairs = list(itertools.product(*[frame[c].unique() for c in by]))
    queries = []
    for values in pairs:
        query = []
        for column, value in zip(by, values):
            if isinstance(value, str):
                query.append(f'(`{column}` == "{value}")')
            else:
                query.append(f"(`{column}` == {value})")
        queries.append("&".join(query))

    sub_frames = L([frame.query(q) for q in queries])
    test_size = int(min(sub_frames.map(len)) * test_perc)
    splits = L(
        [
            train_test_split(
                sub, test_size=test_size, shuffle=shuffle, random_state=random_state
            )
            for sub in sub_frames
        ]
    )
    return pd.concat(splits.itemgot(0)), pd.concat(splits.itemgot(1))


def collect_relevant_kwargs(kwargs, from_callable):
    """Create new dictionary with only relevant kwargs."""
    keys = inspect.signature(from_callable).parameters.keys()
    return {k: kwargs[k] for k in keys}
