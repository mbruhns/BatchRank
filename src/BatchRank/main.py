import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score


def batch_scoring(
    X: pd.DataFrame,
    y,
    clf=HistGradientBoostingClassifier(),
    scoring=balanced_accuracy_score,
    cv=5,
):
    score_lst = []
    label_lst = []

    for label in np.unique(y):
        cv_score = cross_val_score(clf, X.reshape(-1, 1), label, cv=cv)
        score_lst.append(np.median(cv_score))

        score_lst.append(np.median(cv_score))
        label_lst.append(label)

    df = pd.DataFrame({"Label": label_lst, "Score": score_lst})
    df = df.sort_values("Score")

    return df


def main():
    rng = np.random.default_rng(42)

    X = rng.random(size=(1000, 3))
    print(X)
    y = rng.integers(0, 2, size=1000)


if __name__ == "__main__":
    main()
