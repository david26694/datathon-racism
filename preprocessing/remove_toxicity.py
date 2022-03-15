# %%
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm
from detoxify import Detoxify

data_path = Path("data")

# each model takes in either a string or a list of strings
toxify = Detoxify('multilingual')

# optional to display results nicely (will need to pip install pandas)
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")


def batch_predictor(toxify, text, batch_size=32):
    score_list = []

    for i in tqdm.tqdm(range(0, len(text), batch_size)):
        max_idx = min(i + batch_size, len(text))
        output = toxify.predict(text[i:max_idx])
        score_list.append(output)

    return score_list


# %%
text = tweets_raw.text.to_list()

# %%
predictions = batch_predictor(toxify, text)

# %% Aggregate results into dictionary transformable to df
output_dict = defaultdict(list)
for batch in predictions:
    for key, scores in batch.items():
        output_dict[key].extend(scores)
# %%
pd.DataFrame(output_dict).assign(id=tweets_raw.id).to_csv(
    data_path / "detoxify_feats.csv", index=False)
