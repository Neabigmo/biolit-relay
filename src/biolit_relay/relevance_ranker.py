"""Relevance rankers used by the active-learning baselines and BioLit-Relay.

Two rankers share the same interface:

* :class:`TfidfLogRegRanker` - the matched active-learning backbone
  (TF-IDF + Logistic Regression), used by M1/M2 and the relevance component of
  M5 and its ablations.
* :class:`AsreviewDefaultRanker` - ASReview LAB's default simulation
  configuration (Naive Bayes classifier over a TF-IDF document representation,
  ``max`` query strategy), re-instantiated inside the common harness so that
  priors, batch size and budget are identical across methods.

Both are trained only on the currently labelled pool and never see oracle
labels of unlabelled records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_text(records: pd.DataFrame) -> np.ndarray:
    """Title + abstract, whitespace-normalised; missing parts become empty."""
    title = records["title"].fillna("").astype(str).str.strip()
    abstract = records["abstract"].fillna("").astype(str).str.strip()
    return (title + " " + abstract).str.strip().to_numpy()


def fit_tfidf(
    texts: np.ndarray,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    sublinear_tf: bool = True,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit the *shared* TF-IDF representation of a review.

    The representation is fitted once per review on the full candidate pool
    (text only, no labels) and reused by every method so that all methods see
    exactly the same feature space.  Only the classifier is updated online.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        max_features=max_features,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix.tocsr()


class _BaseRanker:
    name = "base"

    def fit_predict(
        self,
        X: sparse.csr_matrix,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        unlabeled_idx: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError


class TfidfLogRegRanker(_BaseRanker):
    """TF-IDF + Logistic Regression (the matched active-learning backbone).

    The design fixes the model family and ``max_iter=1000`` but not the solver;
    ``liblinear`` is used because it is deterministic on sparse TF-IDF features
    and an order of magnitude faster than the default ``lbfgs`` solver at the
    sampled sizes of the larger SYNERGY reviews, which keeps the full 600-run
    simulation reproducible on a CPU-only workstation.
    """

    name = "tfidf_logreg"

    def __init__(self, max_iter: int = 1000, C: float = 1.0, solver: str = "liblinear"):
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.last_fallback = False

    def fit_predict(
        self,
        X: sparse.csr_matrix,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        unlabeled_idx: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        y = np.asarray(labels, dtype=int)
        self.last_fallback = False
        if y.size == 0 or np.unique(y).size < 2:
            self.last_fallback = True
            return np.full(unlabeled_idx.size, 0.5, dtype=float)
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=seed,
        )
        clf.fit(X[labeled_idx], y)
        proba = clf.predict_proba(X[unlabeled_idx])
        classes = list(clf.classes_)
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        return np.asarray(proba[:, pos], dtype=float)


class AsreviewElasU4Ranker(_BaseRanker):
    """ASReview LAB 3.0.8 default active-learning configuration (ELAS u4).

    Source: ``asreview.models.models.ELAS_U4`` (verified in the ASReview 3.0.8
    wheel): maximum querier, TF-IDF feature extractor with word bigrams,
    ``sublinear_tf=True``, ``min_df=1``, ``max_df=0.95``; linear SVM classifier
    with ``loss='squared_hinge'``, ``C=0.11``; and the ``balanced`` balancer with
    ``ratio=9.8`` (class-0 sample weight ``n1 / (ratio * n0)`` relative to 1.0
    for class 1, renormalised to the batch size).

    It is re-instantiated on the *shared* TF-IDF representation used by every
    method in this harness (word 1-2 grams, sublinear TF, ``max_df=1.0``), so
    that priors, batch size, budget accounting and feature space are identical
    across all methods; only the classifier and its class weighting follow the
    ASReview preset.
    """

    name = "asreview_elas_u4_linear_svm"

    def __init__(self, C: float = 0.11, loss: str = "squared_hinge",
                 balance_ratio: float = 9.8, max_iter: int = 5000):
        self.C = C
        self.loss = loss
        self.balance_ratio = balance_ratio
        self.max_iter = max_iter
        self.last_fallback = False

    def fit_predict(
        self,
        X: sparse.csr_matrix,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        unlabeled_idx: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        y = np.asarray(labels, dtype=int)
        self.last_fallback = False
        if y.size == 0 or np.unique(y).size < 2:
            self.last_fallback = True
            return np.full(unlabeled_idx.size, 0.5, dtype=float)
        n1 = float(np.sum(y == 1))
        n0 = float(np.sum(y == 0))
        class0 = n1 / (self.balance_ratio * n0) if n0 > 0 else 1.0
        sample_weight = np.where(y == 1, 1.0, class0)
        sample_weight = sample_weight * (len(y) / sample_weight.sum())
        clf = LinearSVC(C=self.C, loss=self.loss, max_iter=self.max_iter,
                        random_state=seed)
        clf.fit(X[labeled_idx], y, sample_weight=sample_weight)
        scores = clf.decision_function(X[unlabeled_idx])
        return np.asarray(scores, dtype=float).ravel()


# Backward-compatible alias (older scripts referred to the ASReview ranker by
# this name).
AsreviewDefaultRanker = AsreviewElasU4Ranker


@dataclass
class RankerBundle:
    """Convenience container handed to the simulator."""

    X: sparse.csr_matrix
    vectorizer: TfidfVectorizer
    ranker: _BaseRanker = field(default_factory=TfidfLogRegRanker)
