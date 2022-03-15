# %%
"""
Spacy feats, need to run
python -m spacy download es_core_news_sm
before running this script
"""
from pathlib import Path

import numpy as np
import pandas as pd
from whatlies.language._spacy_lang import SpacyLanguage

data_path = Path("data")


# %%
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")
spacy_feats = SpacyLanguage(
    "es_core_news_sm").fit_transform(list(tweets_raw.text))


# %%
n_cols = np.shape(spacy_feats)[1]
col_names = [f"spacy_{i}" for i in range(n_cols)]

spacy_feats_df = pd.DataFrame(
    spacy_feats, columns=col_names).assign(id=tweets_raw.id)

# %% Save
spacy_feats_df.to_csv(
    data_path / "spacy_feats.csv", index=False, float_format="%.3f")
