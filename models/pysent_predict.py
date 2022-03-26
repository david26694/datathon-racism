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


def to_dense(x):
    return x.todense()


# %% Load evaluation data and preprocess
final_name = "evaluation_public"
model_name = "log_reg_pysent"
eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")
pysentimiento_feats = pd.read_csv(
    data_path / "features" / "pysentimiento_evaluation_public.csv").drop(columns=["clean_message"])

eval_df = eval_df_raw.merge(pysentimiento_feats, on=[
                            "message", "label"])

stop_words = get_stop_words('spanish')

eval_df = process_text(eval_df, stop_words)

COLS2DROP = ["message", "label", "processed_msg"]
cols2remain = set(eval_df.columns).difference(COLS2DROP)

X_eval = eval_df.loc[:, cols2remain.union(["processed_msg"])]


# %% Load models and submit
logreg = pickle.load(open(models_path / f"{model_name}.pkl", "rb"))
with open(models_path / f"{model_name}_threshold.txt", "r") as f:
    th_opt = float(f.read())

# %% Create submission file
submission = eval_df_raw.drop(columns=["label"]).assign(
    label=predict_racism(logreg.predict_proba(X_eval), th_opt)
)

submission.to_csv(submission_path / f"{model_name}.csv", index=False, sep="|")

# %%
submission_probs = eval_df_raw.drop(columns=["label"]).assign(
    proba=logreg.predict_proba(X_eval)[:, 1],
)

submission_probs.to_csv(submission_path / f"{model_name}_probs.csv", index=False, sep="|")

# %%
