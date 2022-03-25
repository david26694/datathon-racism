# %% Load libraries and transformer
from pathlib import Path

import pandas as pd
from transformers import pipeline

model_path = "JonatanGk/roberta-base-bne-finetuned-cyberbullying-spanish"
bullying_analysis = pipeline(
    "text-classification", model=model_path, tokenizer=model_path)

data_path = Path("data")

# %% Load data
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")

# %% Apply bullying features
transformed_text = (
    tweets_raw.text
    # Maximum length allowed is 512 tokens, but they are not exaclty words
    .apply(lambda x: x.split()[:100])
    .apply(lambda x: " ".join(x))
    .to_list()
)
bullying_scores = bullying_analysis(transformed_text)
# %% Transform to 0-1 score
bullying_scores_list = [
    x["score"]
    if x["label"] == "Bullying"
    else 1 - x["score"]
    for x in bullying_scores
]
# %% Save features
(
    tweets_raw
    .assign(bullying_score=bullying_scores_list)
    .loc[:, ["id", "bullying_score"]]
    .to_csv(data_path / "bullying_scores.csv", index=False, float_format="%.3f")
)


# %% Check lengths
splitted = tweets_raw.text.apply(lambda s: s.split())
len_split = splitted.apply(lambda s: len(s))
len_split.sort_values()
# %%
