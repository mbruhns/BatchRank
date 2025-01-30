from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import ArrayLike
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def batch_scoring(
    dframe: pd.DataFrame,
    data_columns: ArrayLike,
    label_columns: ArrayLike,
    clf=HistGradientBoostingClassifier(),
    scoring="matthews_corrcoef",
    cv: int = 5,
) -> pd.DataFrame:
    score_lst = []
    label_lst = []

    for label in np.unique(label_columns):
        cv_score = cross_val_score(
            estimator=clf,
            X=dframe[data_columns].values,
            y=dframe[label].values,
            scoring=scoring,
            cv=cv,
        )

        score_lst.append(np.median(cv_score))
        label_lst.append(label)

    df = pd.DataFrame({"Label": label_lst, "Score": score_lst})
    df = df.sort_values("Score", ascending=False)

    return df


def batch_scoring_anndata(
    adata: AnnData,
    data_columns: Union[str, int],
    label_columns: Union[str],
    scoring: Optional[str] = "matthews_corrcoef",
    cv: Optional[int] = 5,
    inplace: bool = True,
) -> Optional[AnnData]:
    X = adata.X
    y = adata.obs[label_columns].values

    # Todo: Implement the actual scoring

    if not inplace:
        return adata


def main():
    rng = np.random.default_rng(42)

    n_samples = 500
    X = rng.random(size=(n_samples, 3))
    pat = rng.integers(0, 5, size=(n_samples, 1))
    lab = rng.integers(0, 3, size=(n_samples, 1))

    test_df = pd.DataFrame(data=X)
    test_df["Patient"] = pat
    test_df["Lab"] = lab

    result = batch_scoring(
        dframe=test_df, data_columns=[0, 1, 2], label_columns=["Patient", "Lab"]
    )
    print(result)


if __name__ == "__main__":
    main()
