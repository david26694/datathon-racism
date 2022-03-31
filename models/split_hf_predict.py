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


model_path = Path("models") / "artifacts" / "hf" / "v1"
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
train_df = pd.read_csv(data_path / "split" /
                       "labels_racism_train.txt", delimiter="|")
test_df = pd.read_csv(data_path / "split" /
                      "labels_racism_test.txt", delimiter="|")


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
train_df = add_max_score(train_df)

# %%
test_df = add_max_score(test_df)

# %%
eval_df_raw = add_max_score(eval_df_raw)

# %%
train_df.head()
# %%

optimal_threshold = safe_threshold_optimisation(
    train_df.max_score,
    (train_df.label == "racist").astype(int),
)

# %%
f1_score(
    (test_df.label == "racist").astype(int),
    test_df.max_score > optimal_threshold,
)

# %%

test_df.to_csv(submission_path / f"{model_name}_validation.csv", index=False)
# %%
(
    test_df
    .query("label == 'racist'")
    .query("max_score < @optimal_threshold")
    .sort_values("max_score", ascending=True)
    .to_csv("tests_david/errors_test.csv", index=False)
)


# %%
(
    test_df
    .query("label == 'non-racist'")
    .query("max_score > @optimal_threshold")
    .sort_values("max_score", ascending=False)
    .to_csv("tests_david/errors_test_2.csv", index=False)
)

# %% Predict and build label

eval_df_raw["score"] = eval_df_raw.max_score
eval_df_raw["label"] = np.where(
    eval_df_raw["score"] > threshold, "racist", "non-racist")
# %% Save data
eval_df_raw.drop(columns=["score", "max_score"]).to_csv(
    submission_path / f"{model_name}.csv", index=False, sep="|")

eval_df_raw.to_csv(submission_path / f"{model_name}_probs.csv", index=False)

# %%
model_name
# %%
test_df
# %%
