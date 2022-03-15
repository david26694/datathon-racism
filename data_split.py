# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = Path("data")
hf_data = Path("hf_data")
spacy_data = Path("spacy_test") / "assets"
hf_data.mkdir(exist_ok=True)


# %% Load data
tweets_raw = pd.read_csv(
    data_path / "hate_speech_data.tsv",
    sep="\t").rename(columns={"HS": "label"})

# %% Split and save (prepare for HF)
train, validation = train_test_split(
    tweets_raw.loc[:, ["text", "label"]], random_state=42)
train.to_csv(hf_data / "train.csv", index=False)
validation.to_csv(hf_data / "validation.csv", index=False)

# %%
pd.concat([
    train.assign(set="train"),
    validation.assign(set="validation")
]).to_csv(spacy_data / "data.csv", index=False)
