# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline

model_path = Path("models") / "artifacts" / "hf" / "v1_full"
data_path = Path("data")
submission_path = Path("data") / "submission"
final_name = "evaluation_public"
model_name = "manual_label"
threshold = 0.34

eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")
labels = pd.read_csv(submission_path / f"labels_manual.txt")
hf_submission = pd.read_csv(submission_path / "hf_v1_full.csv", delimiter="|")


# %%
eval_df_raw["label"] = labels.label
# %%
eval_df_raw.to_csv(submission_path / f"{model_name}.csv", index=False, sep="|")
# %%
hf_submission.merge(eval_df_raw, on="message",
                    how="left").value_counts(["label_x", "label_y"])
# %%
(
    hf_submission
    .merge(eval_df_raw, on="message", how="left", suffixes=("_hf", "_manual")).query("label_hf != label_manual")
    .loc[:, ["label_hf", "label_manual", "message"]]
    .sort_values("label_hf")
    .to_csv("tests_david/test.csv", index=False, sep="|")
)

# %%
hf_submission.merge(eval_df_raw, on="message", how="left",
                    suffixes=("_hf", "_manual")).query("label_x != label_y")

# %%
hf_scores = pd.read_csv(
    submission_path / "hf_v1_full_probs.csv")

# %%
hf_scores.query("score > 0.34").sort_values("score").head(
    10).to_csv("tests_david/test_limit.csv", index=False, sep="|")
# %%
hf_scores.query("score < 0.34").sort_values("score", ascending=False).head(
    10).to_csv("tests_david/test_limit_2.csv", index=False, sep="|")

# %%
