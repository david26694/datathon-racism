
# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline

model_path = Path("models") / "artifacts" / "hf" / "v1_full"
data_path = Path("data")
submission_path = Path("data") / "submission"
final_name = "evaluation_final"
model_name = "hf_v1_full_th_final"
threshold = 0.43

# Load pipeline
p = pipeline("text-classification", model=str(model_path),
             tokenizer=str(model_path))

# %% Load data
eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")

# %% Predict and build label
scores = p(eval_df_raw.message.to_list(), return_all_scores=True)

eval_df_raw["score"] = [x[1]["score"] for x in scores]
eval_df_raw["label"] = np.where(
    eval_df_raw["score"] > threshold, "racist", "non-racist")
# %% Save data
eval_df_raw.drop(columns=["score"]).to_csv(
    submission_path / f"{model_name}.csv", index=False, sep="|")

eval_df_raw.to_csv(submission_path / f"{model_name}_probs.csv", index=False)

# %% Correct with manual data
public_model_submission = pd.read_csv(
    submission_path / "hf_v1_full_th.csv", delimiter="|")
manual_submission = pd.read_csv(
    submission_path / "manual_label.csv", delimiter="|")

# %%
merged = eval_df_raw.merge(manual_submission, how="left", on="message")
merged["label"] = merged.label_y.fillna(merged.label_x)
merged.drop(columns=["score", "label_x", "label_y"]).to_csv(
    submission_path / f"{model_name}_corrected.csv", index=False, sep="|")


# %% Sanity checks
(
    eval_df_raw
    .merge(manual_submission, how="left", on="message")
    .groupby(["label_x", "label_y"], as_index=False)
    .size()
)


# %%
(
    eval_df_raw
    .merge(public_model_submission, how="left", on="message")
    .groupby(["label_x", "label_y"], as_index=False)
    .size()
)

# %% Check differences merged vs final
final = pd.read_csv(
    submission_path / "hf_v1_full_th_final.csv", delimiter="|")
merged_new = pd.read_csv(
    submission_path / "hf_v1_full_th_final_corrected.csv", delimiter="|")


# %%
pd.concat([final.rename(columns={"label": "label_final"}), merged_new], axis=1).groupby(
    ["label_final", "label"], as_index=False).size()
# %%
# valid = pd.read_csv(submission_path / "hf_v1_validation.csv")


# # %% Predict in validation
# p_v1 = pipeline("text-classification", model=str(model_path_v1),
#                 tokenizer=str(model_path_v1))

# # %% Predict
# scores_valid = p(valid["message"].to_list(), return_all_scores=True)

# # %%
# scores_valid_flt = [x[1]["score"] for x in scores_valid]

# # %%
# optimal_cutoff = safe_threshold_optimisation(
#     scores_valid_flt,
#     (valid["label"] == 'racist').astype(int),
# )
# optimal_cutoff

# # %%
# optimal_cutoff = safe_threshold_optimisation(
#     valid["racist_score"],
#     (valid["label"] == 'racist').astype(int),
# )
# optimal_cutoff

# # %% Predict and build label


# def threshold_optimisation(preds, y, n_thresholds=200):

#     df_thresholds = pd.DataFrame()
#     for threshold in np.linspace(0, 1, n_thresholds):
#         df_thresholds = df_thresholds.append(
#             pd.DataFrame({
#                 "threshold": [threshold],
#                 "f1": [f1_score(preds > threshold, y)]
#             })
#         )

#     return df_thresholds.sort_values("f1", ascending=False).head(1).threshold.to_list()[0]


# threshold_optimisation(
#     valid["racist_score"],
#     (valid["label"] == 'racist').astype(int),
# )

# # %% Predict and build label
# threshold_optimisation(
#     scores_valid_flt,
#     (valid["label"] == 'racist').astype(int),
# )
