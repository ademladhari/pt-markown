"""ROC / AUC metrics for watermark detection from Tree-Ring style detector scores.

Lower detector score => closer frequency match => more likely watermarked.
For ROC/AUC we map to classifier scores where *higher* => watermarked.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:
    from sklearn.metrics import auc, roc_curve
except ImportError as e:  # pragma: no cover
    auc = roc_curve = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def detector_scores_to_classifier_scores(detector_scores: Sequence[float]) -> np.ndarray:
    """Convert detector distance (lower = watermarked) to score (higher = watermarked)."""
    x = np.asarray(detector_scores, dtype=np.float64)
    return -x


def roc_watermark_vs_clean(
    detector_scores_watermarked: Sequence[float],
    detector_scores_clean: Sequence[float],
) -> Dict[str, float | np.ndarray]:
    """Binary ROC with label 1 = watermarked, 0 = clean.

    Returns auc, arrays fpr/tpr/thresholds, and counts for plotting.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "roc_watermark_vs_clean requires scikit-learn. Install with: pip install scikit-learn"
        ) from _IMPORT_ERROR

    y_wm = np.ones(len(detector_scores_watermarked), dtype=np.int32)
    y_cl = np.zeros(len(detector_scores_clean), dtype=np.int32)
    y_true = np.concatenate([y_wm, y_cl])
    det = np.asarray(list(detector_scores_watermarked) + list(detector_scores_clean), dtype=np.float64)
    y_score = detector_scores_to_classifier_scores(det)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need at least one positive and one negative sample for ROC.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    return {
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "n_watermarked": int(len(detector_scores_watermarked)),
        "n_clean": int(len(detector_scores_clean)),
        "mean_score_wm": float(np.mean(detector_scores_watermarked)),
        "mean_score_clean": float(np.mean(detector_scores_clean)),
    }


def format_roc_report(result: Dict) -> str:
    lines = [
        f"ROC AUC (watermarked vs clean): {result['auc']:.4f}",
        f"n_watermarked={result['n_watermarked']}, n_clean={result['n_clean']}",
        f"mean detector score — wm={result['mean_score_wm']:.4f}, "
        f"clean={result['mean_score_clean']:.4f} (lower is stronger watermark signal)",
    ]
    return "\n".join(lines)
