from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


def ward_cluster(beta_hat: pd.DataFrame, n_clusters: int) -> pd.Series:
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if beta_hat.empty:
        return pd.Series([], dtype=int, index=beta_hat.index, name="cluster")
    if beta_hat.isna().any().any():
        raise ValueError("beta_hat contains NaN")
    data = beta_hat.to_numpy()
    z = linkage(data, method="ward")
    labels = fcluster(z, t=n_clusters, criterion="maxclust")
    return pd.Series(labels, index=beta_hat.index, name="cluster")


def choose_n_eff(labels: pd.Series, m_min: int) -> int:
    if m_min < 1:
        raise ValueError("m_min must be >= 1")
    counts = labels.value_counts()
    return int((counts >= m_min).sum())


def cluster_centroids(beta_hat: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    if len(beta_hat) != len(labels):
        raise ValueError("labels length must match beta_hat rows")
    centroids = beta_hat.groupby(labels).mean()
    centroids.index.name = "cluster"
    return centroids


def select_representative(
    cluster: pd.DataFrame,
    S_fit: pd.Series,
    centroid: np.ndarray,
    eta: float,
) -> str:
    if cluster.empty:
        raise ValueError("cluster is empty")
    if eta < 0:
        raise ValueError("eta must be >= 0")
    scores = S_fit.reindex(cluster.index)
    if scores.isna().any():
        raise ValueError("S_fit missing for cluster")
    distances = np.linalg.norm(cluster.to_numpy() - centroid.reshape(1, -1), axis=1)
    adjusted = scores.to_numpy() - eta * distances
    order = np.lexsort((cluster.index.astype(str).to_numpy(), -adjusted))
    return str(cluster.index[order[0]])


def topk_candidates(
    cluster: Sequence[str] | pd.Index,
    score_J: pd.Series,
    k: int,
) -> list[str]:
    if k <= 0:
        return []
    idx = pd.Index(cluster)
    scores = score_J.reindex(idx)
    if scores.isna().any():
        raise ValueError("score_J missing for cluster")
    ordered = sorted(idx, key=lambda code: (-scores[code], str(code)))
    return [str(code) for code in ordered[:k]]
