# %%
"""Load hate speech analyzer and get features for tweet text"""


from pathlib import Path

import pandas as pd
from pysentimiento import create_analyzer

from utils.analyzer_utils import get_analyzer_features

hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")

data_path = Path("data")
data_path.mkdir(exist_ok=True)

# %% Some examples
# hate_speech_analyzer.predict("Me gustan los inmigrantes pero deben ser aniquilados")
# hate_speech_analyzer.predict("Yo no soy racista, soy ordenado")
# hate_speech_analyzer.predict("Conduce como un chino")

# %% Load data
# https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-A/train_es.tsv
tweets = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")

# %% Hat speech features from analyzer
# hate_feats = get_analyzer_features(tweets.text.head(100), hate_speech_analyzer)

# %% All features
df = tweets.copy().reset_index(drop=True)
feats_dict = {}
for task in ["hate_speech", "emotion", "sentiment"]:
    analyzer = create_analyzer(task=task, lang="es")
    feats_dict[task] = get_analyzer_features(df.text, analyzer)


all_feats = pd.concat(feats_dict.values(), axis=1)
final_df = pd.concat([df, all_feats], axis=1)

# %% Save data
(
    final_df
    .drop(columns=["text", "HS", "TR", "AG"])
    .to_csv(data_path / "pysentimiento_feats.csv", index=False, float_format="%.3f")
)
