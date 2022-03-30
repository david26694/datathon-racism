# %% Load libraries and pipeline
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import pipeline
from utils.thresholds import safe_threshold_optimisation


def split_sentence(x):
    splits = x.split(".")
    return [x for x in splits if len(x) > 15]


model_path = Path("models") / "artifacts" / "hf" / "v1_full"
data_path = Path("data")
submission_path = Path("data") / "submission"
final_name = "evaluation_public"
model_name = "hf_v1_split"
threshold = 0.45

# Load pipeline
p = pipeline("text-classification", model=str(model_path),
             tokenizer=str(model_path))

# %% Load data
eval_df_raw = pd.read_csv(data_path / f"{final_name}.csv", delimiter="|")

# %%


def add_max_score(df):
    df = df.copy()
    df["max_score"] = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        many_sentences = split_sentence(str(row["message"]))
        score = p(many_sentences, return_all_scores=True)
        max_score = max(score, key=lambda x: x[1]["score"])[1]["score"]
        df.loc[i, "max_score"] = max_score
    return df


# %%
eval_df_raw = add_max_score(eval_df_raw)

# %% Predict and build label

eval_df_raw["score"] = eval_df_raw.max_score
eval_df_raw["label"] = np.where(
    eval_df_raw["score"] > threshold, "racist", "non-racist")
# %% Save data
eval_df_raw.drop(columns=["score", "max_score"]).to_csv(
    submission_path / f"{model_name}.csv", index=False, sep="|")

eval_df_raw.to_csv(submission_path / f"{model_name}_probs.csv", index=False)

# %%
