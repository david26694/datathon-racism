# %% Load libraries and transformer
from pathlib import Path

import pandas as pd
from transformers import pipeline

model_path = "Hate-speech-CNERG/dehatebert-mono-spanish"
dehate_pipe = pipeline(
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
dehate_scores = dehate_pipe(
    transformed_text,
    batch_size=32,
    truncation="do_not_truncate",
    return_all_scores=True
)
# %%
rows = []
for row in dehate_scores:
    d = {}
    for emotion in row:
        d[emotion["label"]] = emotion["score"]
    rows.append(d)

dehate_df = pd.DataFrame(rows)

# %% Save features
(
    dehate_df
    .assign(id=tweets_raw.id)
    .to_csv(data_path / "dehatebert_feats.csv", index=False, float_format="%.3f")
)


# %% Check lengths
splitted = tweets_raw.text.apply(lambda s: s.split())
len_split = splitted.apply(lambda s: len(s))
len_split.sort_values()
# %%
