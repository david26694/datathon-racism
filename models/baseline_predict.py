# %%
"""
Sklearn baseline
"""
import pickle
from pathlib import Path

import pandas as pd
from stop_words import get_stop_words
from utils.classic_preprocessing import process_text
from utils.thresholds import predict_racism

# Make sure you have these folders
data_path = Path("data")
submission_path = data_path / "submission"
models_path = Path("models") / "artifacts"


# %% Load evaluation data and preprocess
final_name = "evaluation_public"
eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")

stop_words = get_stop_words('spanish')

eval_df = process_text(eval_df_raw, stop_words)

X_eval = eval_df.processed_msg

# %% Load models and submit
logreg = pickle.load(open(models_path / "log_reg_baseline.pkl", "rb"))
with open(models_path / "log_reg_baseline_threshold.txt", "r") as f:
    th_opt = float(f.read())

# %% Create submission file
submission = eval_df_raw.drop(columns=["label"]).assign(
    label=predict_racism(logreg.predict_proba(X_eval), th_opt)
)

submission.to_csv(submission_path / "log_reg_baseline.csv", index=False)

# %%
