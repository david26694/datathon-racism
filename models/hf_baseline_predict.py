
# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline

model_path = Path("models") / "artifacts" / "hf" / "v1_full"
data_path = Path("data")
submission_path = Path("data") / "submission"
final_name = "evaluation_public"
model_name = "hf_v1_full"
threshold = 0.5

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
    submission_path / f"{model_name}.csv", index=False)

eval_df_raw.to_csv(submission_path / f"{model_name}_probs.csv", index=False)

# %%
