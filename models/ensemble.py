
# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from utils.thresholds import safe_threshold_optimisation

data_path = Path("data")
submission_path = Path("data") / "submission"

file_1 = "hf_regress_validation.csv"
file_2 = "hf_v1_validation.csv"
cutoff = 0.44

# %%
val_1 = pd.read_csv(submission_path / file_1)
val_2 = pd.read_csv(submission_path / file_2)
# %%
y = (val_1.label == 'racist').astype(int)
# %%
f1_score(y, val_1.racist_score > cutoff)
# %%
f1_score(y, val_2.racist_score > cutoff)
# %%
f1_score(y, (val_1.racist_score + val_2.racist_score) / 2 > cutoff)
# %%
# Weighted mix
# Normalise 0-1 with ranks and mix
qt = QuantileTransformer(n_quantiles=1e4, random_state=0)
val_1["normalised_scores"] = qt.fit_transform(val_1.loc[:, ["racist_score"]])
val_2["normalised_scores"] = qt.fit_transform(val_2.loc[:, ["racist_score"]])

# %%
normalised_mix = (val_1.normalised_scores + val_2.normalised_scores) / 2
# %%
new_cutoff = qt.transform([[cutoff]])[0][0]
new_cutoff
# %%
f1_score(y, normalised_mix > new_cutoff)

# %%
opti_cutoff = safe_threshold_optimisation(normalised_mix, y)
f1_score(y, normalised_mix > opti_cutoff)
# %%
for w in np.linspace(0, 1, 100):
    normalised_mix = (val_1.normalised_scores * (1 - w) +
                      val_2.normalised_scores * w)
    opti_cutoff = safe_threshold_optimisation(normalised_mix, y)
    print(w, opti_cutoff, f1_score(y, normalised_mix > opti_cutoff))

# %%
