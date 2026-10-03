"""Classification scores for burn (1) versus coast (0) predictions."""
import numpy as np


def score_predictions(predicted, truth):
    """
    Confusion counts plus sensitivity (recall), specificity, precision and balanced accuracy.

    Balanced accuracy is the honest headline number here: burns are well under 5% of the
    samples, so "always predict coast" already scores 95%+ raw accuracy.
    """
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sensitivity=sensitivity, specificity=specificity,
                precision=precision, balanced_accuracy=0.5 * (sensitivity + specificity))


def balanced_accuracy(predicted, truth):
    return score_predictions(predicted, truth)["balanced_accuracy"]


def accuracy(predicted, truth):
    return float(np.mean(predicted == truth))
