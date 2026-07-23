"""Open-set rejection via Mahalanobis distance to per-class prototypes.

The 6-class softmax is closed-world: it always picks one of the 6, confidently,
even for a wrong or mixed sign. This adds an "is it a known sign at all?" check
in the model's embedding space (the layer before softmax): fit a prototype
(mean) per class plus a shared covariance, then at inference measure the
Mahalanobis distance to the nearest prototype. An input far from every cluster
is flagged 'unknown' regardless of what softmax says (Lee et al., 2018).
"""

import numpy as np


def embedding_model(model):
    """Keras sub-model producing the penultimate (pre-softmax) features."""
    from tensorflow import keras
    return keras.Model(model.inputs, model.layers[-2].output)


def fit_prototypes(Z, y, n_classes):
    """Z (n,d) embeddings, y (n,) labels -> (means (k,d), precision (d,d)).
    Shared within-class covariance with Ledoit-Wolf shrinkage (stable for small n)."""
    from sklearn.covariance import LedoitWolf
    means = np.stack([Z[y == c].mean(axis=0) for c in range(n_classes)]).astype(np.float64)
    centered = np.concatenate([Z[y == c] - means[c] for c in range(n_classes)], axis=0)
    precision = LedoitWolf().fit(centered).precision_.astype(np.float64)
    return means, precision


def mahalanobis_min(Z, means, precision):
    """Return (min_dist (n,), argmin_class (n,)) over prototypes for each row."""
    Z = np.asarray(Z, dtype=np.float64)
    d2 = np.empty((Z.shape[0], means.shape[0]))
    for c in range(means.shape[0]):
        diff = Z - means[c]
        d2[:, c] = np.einsum("ij,jk,ik->i", diff, precision, diff)
    dists = np.sqrt(np.clip(d2, 0, None))
    return dists.min(axis=1), dists.argmin(axis=1)


def save_stats(path, means, precision, threshold, percentile):
    np.savez(path, means=means, precision=precision,
             threshold=np.float64(threshold), percentile=np.float64(percentile))


def load_stats(path):
    d = np.load(path)
    return d["means"], d["precision"], float(d["threshold"]), float(d["percentile"])
