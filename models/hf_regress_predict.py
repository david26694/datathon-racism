
# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from pysentimiento.preprocessing import preprocess_tweet
from transformers import pipeline

model_path = Path("models") / "artifacts" / "hf" / "regression-model"
data_path = Path("data")
submission_path = Path("data") / "submission"
final_name = "evaluation_public"
model_name = "hf_regress"
threshold = 0.44

# Load pipeline
p = pipeline("text-classification", model=str(model_path),
             tokenizer=str(model_path))

# %% Load data
eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")
eval_df_raw = eval_df_raw.assign(
    text=lambda x: x.message.apply(preprocess_tweet))

# %% Predict and build label
scores = p(eval_df_raw.text.to_list(), return_all_scores=True)

# %%
eval_df_raw["score"] = [x[0]["score"] for x in scores]
eval_df_raw["label"] = np.where(
    eval_df_raw["score"] > threshold, "racist", "non-racist")
# %% Save data
eval_df_raw.drop(columns=["score", "text"]).to_csv(
    submission_path / f"{model_name}.csv", index=False, sep="|")

eval_df_raw.to_csv(submission_path / f"{model_name}_probs.csv", index=False)

# %%
