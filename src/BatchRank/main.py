from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def batch_scoring(
    dframe: pd.DataFrame,
    data_columns: Union[List[str], Sequence[str]],
    label_columns: Union[List[str], Sequence[str]],
    clf: Optional[object] = None,
    scoring: str = "matthews_corrcoef",
    cv: int = 5,
) -> pd.DataFrame:
    """
    Compute median cross-validated scores for each label column in a DataFrame.

    Parameters
    ----------
    dframe : pd.DataFrame
        DataFrame containing both predictor and label columns.
    data_columns : list of str or sequence of str
        Column names to use as features.
    label_columns : list of str or sequence of str
        Column names to use as target labels.
    clf : estimator, optional
        Classifier to use for scoring. Defaults to HistGradientBoostingClassifier.
    scoring : str, default "matthews_corrcoef"
        Scoring metric to use.
    cv : int, default 5
        Number of folds in cross-validation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "Label" and "Score", sorted by Score in descending order.
    """
    if clf is None:
        clf = HistGradientBoostingClassifier()

    # Validate that the provided data and label columns exist in the DataFrame.
    missing_features = set(data_columns) - set(dframe.columns)
    if missing_features:
        raise ValueError(
            f"The following feature columns are missing: {missing_features}"
        )

    missing_labels = set(label_columns) - set(dframe.columns)
    if missing_labels:
        raise ValueError(
            f"The following label columns are missing: {missing_labels}"
        )

    # Precompute the feature matrix once.
    X = dframe.loc[:, data_columns].values

    # Use np.unique in case duplicate labels were provided.
    results = []
    for label in np.unique(label_columns):
        y = dframe[label].values
        cv_scores = cross_val_score(
            estimator=clf, X=X, y=y, scoring=scoring, cv=cv
        )
        results.append((label, np.median(cv_scores)))

    # Build and sort the results DataFrame.
    result_df = pd.DataFrame(results, columns=["Label", "Score"])
    result_df.sort_values("Score", ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    return result_df


def batch_scoring_anndata(
    adata: AnnData,
    feature_keys: Optional[Union[List[str], str]] = None,
    label_keys: Union[List[str], str] = None,
    feature_source: str = "X",
    clf: Optional[object] = None,
    scoring: str = "matthews_corrcoef",
    cv: int = 5,
) -> pd.DataFrame:
    """
    Perform batch scoring via cross-validation for specified labels on an AnnData object,
    using a chosen feature source (e.g. 'X' or a layer in adata.layers).

    For each label (i.e. column in `adata.obs`) provided in `label_keys`, this function
    evaluates the provided classifier using cross-validation with predictors defined by
    `feature_keys` (or all features if None) extracted from the specified data source,
    and returns a DataFrame of median scores sorted in descending order.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell data.
    feature_keys : list of str or str, optional
        Feature(s) (e.g. gene names) to use as predictors. If None, all features are used.
    label_keys : list of str or str
        Label column name(s) in `adata.obs` to use as targets.
    feature_source : str, default "X"
        The source of the feature data. Use "X" to select `adata.X` or provide a key that is
        present in `adata.layers` to use that layer.
    clf : classifier, optional
        Scikit-learn classifier to use. Defaults to HistGradientBoostingClassifier if None.
    scoring : str, default "matthews_corrcoef"
        Scoring metric used for cross-validation.
    cv : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "Label" and "Score" containing the median cross-validation
        score for each label, sorted by "Score" in descending order.

    Raises
    ------
    ValueError
        If `label_keys` is not provided, if any provided keys are not found in the AnnData
        object, or if `feature_source` is invalid.
    """
    if clf is None:
        clf = HistGradientBoostingClassifier()

    # Validate and subset features
    if feature_keys is not None:
        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]
        missing_features = set(feature_keys) - set(adata.var_names)
        if missing_features:
            raise ValueError(
                f"The following features are not found in adata.var_names: {missing_features}"
            )
        # Subset the AnnData object to the selected features
        adata_subset = adata[:, feature_keys]
    else:
        adata_subset = adata

    # Select the feature matrix from the specified source.
    if feature_source == "X":
        X = adata_subset.X
    elif feature_source in adata_subset.layers:
        X = adata_subset.layers[feature_source]
    else:
        raise ValueError(
            f"The feature_source '{feature_source}' is not found in adata.layers "
            "and is not 'X'."
        )

    # Validate labels
    if label_keys is None:
        raise ValueError(
            "You must provide at least one label key via the `label_keys` parameter."
        )
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    missing_labels = set(label_keys) - set(adata.obs.columns)
    if missing_labels:
        raise ValueError(
            f"The following label keys are not found in adata.obs: {missing_labels}"
        )

    scores = []
    labels = []

    # Evaluate each label
    for label in np.unique(label_keys):
        y = adata.obs[label].values
        cv_scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
        scores.append(np.median(cv_scores))
        labels.append(label)

    result_df = pd.DataFrame({"Label": labels, "Score": scores})
    result_df.sort_values("Score", ascending=False, inplace=True)
    return result_df


def main():
    """
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
    """

    # Create a dummy AnnData object with a layer named 'raw'
    n_cells, n_genes = 100, 10
    adata = AnnData(
        X=np.random.rand(n_cells, n_genes),
        obs=pd.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B"], size=n_cells),
                "condition": np.random.choice(
                    ["treated", "control"], size=n_cells
                ),
            }
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    # For demonstration, create a 'raw' layer
    adata.layers["raw"] = adata.X * 2  # Example transformation

    # Perform batch scoring using features from the 'raw' layer
    result = batch_scoring_anndata(
        adata,
        feature_keys=["gene_1", "gene_2", "gene_3"],
        label_keys=["cell_type", "condition"],
        feature_source="raw",
    )
    print(result)


if __name__ == "__main__":
    main()
