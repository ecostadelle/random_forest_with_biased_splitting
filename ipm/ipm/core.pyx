import numpy as np
cimport numpy as cnp
cnp.import_array()

from libcpp.unordered_set cimport unordered_set
from libcpp.set cimport set
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc cimport math

from sklearn.tree._tree cimport Tree
from sklearn.tree._tree cimport Node
from sklearn.tree._utils cimport rand_int, rand_uniform

from sklearn.ensemble._forest import _generate_sample_indices


ctypedef   Py_ssize_t intp_t
ctypedef       double float64_t
ctypedef   signed int int32_t
ctypedef unsigned int uint32_t


cdef vector[intp_t] _node_indices_in_path(
    Node* nodes, 
    float64_t[:,:] X,
    intp_t i,
    intp_t max_depth
) noexcept nogil:
    cdef:
        intp_t j, k=0
        vector[intp_t] path
    # wile it isn't a leaf
    path.reserve(max_depth)
    while nodes[k].feature > -1:
        j = nodes[k].feature
        # add the split node id to path
        path.emplace_back(k)
        if X[i,j] <= nodes[k].threshold:
            k = nodes[k].left_child
        else:
            k = nodes[k].right_child
    # finally, add the leaf node id to path
    path.emplace_back(k)
    return path


cdef vector[intp_t] _leaf_node_in_path(
    Node* nodes, 
    float64_t[:,:] X,
    intp_t i,
    intp_t max_depth
) noexcept nogil:

    return _node_indices_in_path(nodes, X, i, max_depth)


cdef intp_t get_leaf_node_in_path(
    object decision_tree, 
    float64_t[:,:] X,
    intp_t i
):
    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes

    return _node_indices_in_path(nodes, X, i, tree.max_depth)[-1]


cdef vector[intp_t] _features_vector_in_path(
    Node* nodes, 
    float64_t[:,:] X,
    intp_t i,
    intp_t max_depth
) noexcept nogil:
    cdef:
        vector[intp_t] features_vector
        intp_t feature, node

    features_vector.reserve(max_depth)
    for node in _node_indices_in_path(nodes, X, i, max_depth):
        feature = nodes[node].feature
        if feature >= 0 :
            features_vector.emplace_back(feature)

    return features_vector

cdef set[intp_t] _features_set_in_path(
    Node* nodes, 
    float64_t[:,:] X,
    intp_t i,
    intp_t max_depth
) noexcept nogil:
    cdef:
        set[intp_t] features_set
        intp_t feature

    for feature in _features_vector_in_path(nodes, X, i, max_depth):
        features_set.insert(feature)

    return features_set


cpdef cnp.ndarray get_node_indices_in_path(
    object decision_tree, 
    float64_t[:,:] X,
    intp_t i
):
    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        
    return np.asarray(_node_indices_in_path(nodes, X, i, tree.max_depth))

cpdef cnp.ndarray get_features_vector_in_path(
    object decision_tree, 
    float64_t[:,:] X,
    intp_t i
):
    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        vector[intp_t] features_vector = _features_vector_in_path(nodes, X, i, tree.max_depth)

    return np.asarray(features_vector)


cpdef object get_features_set_in_path(
    object decision_tree, 
    float64_t[:,:] X,
    intp_t i
):
    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        set[intp_t] features_set = _features_set_in_path(nodes, X, i, tree.max_depth)

    return features_set


cpdef cnp.ndarray _ipm(object decision_tree, float64_t[:,:] X): 
    """
    Compute the Intervention in Prediction Measure (IPM) as proposed by Epifânio (2017).

    This implementation follows the original definition, where all occurrences 
    of features along a decision path are counted (i.e., multiple appearances 
    of the same feature are allowed and accumulated). The contribution of each 
    feature is weighted by the inverse of the path length and normalized by the 
    number of samples.

    Parameters
    ----------
    decision_tree : object
        A trained decision tree object (e.g., from scikit-learn). The function 
        accesses the underlying structure via the `tree_` attribute.

    X : ndarray of shape (n_samples, n_features)
        Input feature matrix used to traverse the tree and compute 
        feature-level contributions.

    Returns
    -------
    ipm : ndarray of shape (n_features,)
        An array representing the importance measure for each feature. For each 
        sample, every occurrence of a feature in the decision path adds a 
        proportional value (1 / path_length / n_samples) to its corresponding 
        position in the result.

    Notes
    -----
    - This function traverses the decision tree for each sample in `X` and records every feature used in the decision path, including duplicates.
    - No filtering is performed to ensure uniqueness; features that appear multiple times 
      in the path contribute multiple times to the final score.
    - The result is normalized by the number of samples and the length of each path, 
      preserving the original formulation of Epifânio (2017).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from ipm import _ipm
    >>> X = np.array([
    ...     [1.0, 2.0],
    ...     [3.0, 4.0],
    ...     [5.0, 6.0],
    ...     [7.0, 8.0]
    ... ])
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    >>> ipm = _ipm(clf, X[[1]])
    >>> print(ipm)
    [0.5, 0.5]  # If both features appear once on the path of X[1]
    """

    cdef:
        Tree tree = <Tree> decision_tree.tree_
        Node* nodes = <Node*> tree.nodes
        vector[intp_t] features_vector
        intp_t i, j, k
        intp_t n = <intp_t> X.shape[0]
        intp_t m = <intp_t> X.shape[1]
        float64_t[:] ipm = np.zeros(m)

    with nogil:
        for i in range(n):
            features_vector = _features_vector_in_path(nodes, X, i, 
                                                       tree.max_depth)
            
            for feature in features_vector:
                ipm[feature] += 1/features_vector.size()/n

    return np.asarray(ipm)



def get_oob_idx(decision_tree, n, max_samples):
    """
    Retrieve out-of-bag (OOB) indices for a given decision tree.

    Parameters:
    -----------
    decision_tree : object
        Decision tree object.
    n : int
        Total number of samples.
    max_samples : float or int
        The maximum number of samples for bootstrapping.

    Returns:
    --------
    ndarray
        Indices of OOB samples.
    """

    random_state = decision_tree.random_state
    bag_idx = _generate_sample_indices(random_state, n, max_samples)
    sample_counts = np.bincount(bag_idx, minlength=n)
    oob_idx = np.arange(n, dtype=np.intp)[sample_counts==0]

    return oob_idx