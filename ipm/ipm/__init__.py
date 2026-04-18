import numpy as np
import pandas as pd

from joblib.parallel import (
    Parallel, 
    delayed,
)

from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_sample_indices

from .core import _ipm

from typing import Union


def ipm(decision_tree:DecisionTreeClassifier, X:np.ndarray): 
    """
    Computes the Importance in Prediction Measure (IPM) for each 
    feature in the dataset based on the given decision tree.

    Parameters
    ----------
    decision_tree : DecisionTreeClassifier
        A trained decision tree object, typically from scikit-learn. The function accesses 
        the underlying tree structure via the `tree_` attribute.

    X : ndarray of shape (n_samples, n_features), dtype=float64
        The input feature matrix for which the IPM is to be computed.

    Returns
    -------
    ipm : ndarray of shape (n_features,)
        An array representing the importance measure of each feature. The importance is 
        computed based on the number of times a feature is encountered across all paths 
        in the decision tree, normalized by the number of samples and the number of features 
        involved in each path.

    Notes
    -----
    - The function works by traversing the decision tree for each sample in `X` and 
      accumulating the feature contributions across the decision paths.
    - The contribution of each feature is computed as the inverse of the number of unique 
      features in the path, divided by the total number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from ipm import ipm
    >>> X = np.array([
    ...     [1.0, 2.0], 
    ...     [3.0, 4.0], 
    ...     [5.0, 6.0], 
    ...     [7.0, 8.0]
    ... ])
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=3).fit(X, y)

    The resulting tree structure:
    
    For the input sample X[1] = [3.0, 4.0], the decision path traverses:
      - x[0] (one time),
      - x[1] (one time, even though it appears twice in the tree).

    The expected output, considering unique feature contributions:

    >>> ipm_result = ipm(clf, X[[1]])
    >>> print(ipm_result)
    [0.5, 0.5] 

    """

    return _ipm(decision_tree, X)


def check_data(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.values.astype(np.float64)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        print(data.dtype)
        raise TypeError('Input data must be a Pandas DataFrame, Series or a NumPy ndarray')


def _parallel_ipm(forest: RandomForestClassifier, X: np.ndarray, n_jobs: int):
    """
    Parallel computation of IPM for a RandomForestClassifier.

    Args:
        forest (RandomForestClassifier): Trained RandomForestClassifier.
        X (np.ndarray): Input data.
        n_jobs (int): Number of parallel jobs.

    Returns:
        np.ndarray: IPM values for each feature.
    """
    result = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_ipm)
        (
            decision_tree,
            X,
        ) for decision_tree in forest.estimators_
    )

    result = np.mean(result, axis=0)

    # L1-normalization is necessary because some trees can got qipm equal zero
    result /= result.sum()

    return result


def get_ipm(forest: RandomForestClassifier, 
            X: Union[pd.DataFrame, np.ndarray],
            n_jobs: int = -1):
    """
    Compute the Intervention in Prediction Measure (IPM) for a given 
    RandomForestClassifier.

    This function estimates the IPM for each feature by analyzing the decision 
    paths followed by instances in the input data across all trees in the 
    forest.

    Parameters
    ----------
    forest : RandomForestClassifier
        A trained RandomForestClassifier model.

    X : {pd.DataFrame, np.ndarray} of shape (n_samples, n_features)
        The input feature matrix. Each row corresponds to an instance.

    n_jobs : int, default=-1
        The number of parallel jobs to run. If -1, uses all available processors.

    Returns
    -------
    ipm : np.ndarray
        - If `per_instance` is False: array of shape (n_features,), representing the global
          IPM vector, normalized to sum to 1 (L1-normalized).
        - If `per_instance` is True: array of shape (n_samples, n_features), where each row
          contains the IPM vector for the corresponding instance.

    Notes
    -----
    - The IPM is computed by identifying the features used in the decision path of each 
      instance in each tree and assigning them equal importance inversely proportional to 
      the number of features in that path.
    - This implementation uses parallel computation across instances.
    """
    X = check_data(X)

    return _parallel_ipm(forest, X, n_jobs)