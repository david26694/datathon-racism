import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def threshold_optimisation(preds, y, n_thresholds=200):

    df_thresholds = pd.DataFrame()
    for threshold in np.linspace(0, 1, n_thresholds):
        df_thresholds = df_thresholds.append(
            pd.DataFrame({
                "threshold": [threshold],
                "f1": [f1_score(preds[:, 1] > threshold, y)]
            })
        )

    return df_thresholds.sort_values("f1", ascending=False).head(1).threshold.to_list()[0]


def safe_threshold_optimisation(preds, y, n_thresholds=200, acceptance_cutoff=0.003):

    df_thresholds = pd.DataFrame()
    for threshold in np.linspace(0, 1, n_thresholds):
        df_thresholds = df_thresholds.append(
            pd.DataFrame({
                "threshold": [threshold],
                "f1": [f1_score(preds > threshold, y)]
            })
        )
    max_f1 = df_thresholds.f1.max()
    # Most safe threshold
    return df_thresholds[(max_f1 - df_thresholds["f1"]) < acceptance_cutoff].threshold.median()


def predict_racism(probas, threshold):
    return np.where(probas[:, 1] >= threshold, "racist", "non-racist")
