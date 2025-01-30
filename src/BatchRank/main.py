import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score


def batch_scoring(
    X: pd.DataFrame,
    data_columns,
    label_columns,
    clf=HistGradientBoostingClassifier(),
    scoring=balanced_accuracy_score,
    cv=5,
):
    score_lst = []
    label_lst = []

    for label in np.unique(label_columns):
        cv_score = cross_val_score(
            clf, X[data_columns].values, X[label].values, cv=cv
        )
        score_lst.append(np.median(cv_score))
        label_lst.append(label)

    df = pd.DataFrame({"Label": label_lst, "Score": score_lst})
    df = df.sort_values("Score", ascending=False)

    return df


def main():
    rng = np.random.default_rng(42)

    X = rng.random(size=(1000, 3))
    y = rng.integers(0, 2, size=(1000, 2))

    test_df = pd.DataFrame(data=X)
    test_df["A"] = y[:, 0]
    test_df["B"] = y[:, 1]

    result = batch_scoring(
        X=test_df, data_columns=[0, 1, 2], label_columns=["A", "B"]
    )
    print(result)


if __name__ == "__main__":
    main()
