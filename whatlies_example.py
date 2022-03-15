# %%
"""
Pipeline example for the whatlies library with HF model."""
from pathlib import Path

import numpy as np
import pandas as pd
import pip
import spacy
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from whatlies.language import HFTransformersLanguage
from whatlies.language._spacy_lang import SpacyLanguage

from config import PYSENTIMIENTO_FEATS
from utils.misc import ToListTransformer

data_path = Path("data")


# %% Load data from HF
# lang = HFTransformersLanguage("pysentimiento/robertuito-sentiment-analysis")

# Example of transformation
# lang.fit_transform(['eres muy majo', 'eres muy malo'])

grab_text_transform = ("grab_text", ColumnTransformer(
    [("selector", "passthrough", ["text"])]))

# Create logreg pipeline and train
pipe = Pipeline([
    ("ml_features", FeatureUnion([
        # ("transformer_pipeline", Pipeline([
        #     ("grab_text", ColumnTransformer(
        #         [("selector", "passthrough", ["text"])])),
        #     ("to_list", ToListTransformer()),
        #     ("embed", HFTransformersLanguage(
        #         "pysentimiento/robertuito-sentiment-analysis"))
        # ])),
        # ("counts_pipeline", Pipeline([
        #     ("grab_text", ColumnTransformer([("selector", "passthrough", ["text"])])),
        #     ("to_list", ToListTransformer()),
        #     ("counts", CountVectorizer(max_features=20, stop_words='spanish'))
        # ])),
        ("spacy_pipeline", Pipeline([
            ("grab_text", ColumnTransformer(
                [("selector", "passthrough", ["text"])])),
            ("to_list", ToListTransformer()),
            ("spacy", SpacyLanguage("es_core_news_sm"))
        ])),
        ("grab_pysentimiento", ColumnTransformer([
            ("selector", "passthrough", PYSENTIMIENTO_FEATS)
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])


# %% Load data
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")
pysentimiento_feats = pd.read_csv(data_path / "pysentimiento_feats.csv")

# %% Preprocessing
tweets = tweets_raw.merge(pysentimiento_feats, on="id", how="left")

X = tweets.drop(columns=["id", "HS", "TR", "AG"])
y = tweets.HS

# %%
# lang = HFTransformersLanguage("pysentimiento/robertuito-sentiment-analysis")
# lang.fit_transform(list(X.text))

# spacy_feats = SpacyLanguage("es_core_news_sm").fit_transform(list(X.text))

# %% Model training
pipe.fit(X, y)


# %% Model evaluation
cross_val_score(pipe, X, y, cv=5, scoring="average_precision").mean()

# %%
y_diff = np.abs(pipe.predict_proba(X)[:, 1] - y)

tweets.assign(y_diff=y_diff, prediction=pipe.predict_proba(X)
              [:, 1]).sort_values("y_diff", ascending=False).head(10).text.to_list()

# %% Toy model

# X = ['eres muy majo', 'eres muy malo']

# y = np.array([1, 0])

# pipe.fit(X, y)

# pipe.predict_proba([
#     'eres malisimo',
#     'eres lo peor',
#     'eres lo mejor',
#     'me caes bien'
# ])

# pipe.predict_proba(['eres lo peor'])

# pipe.predict_proba(['eres lo mejor'])


# %%
# p = Pipeline([
#     grab_text_transform,
#     ("to_list", ToListTransformer()),
#     ("counts", CountVectorizer(max_features=20, stop_words='english'))
# ])
# p.fit(X, y)
# p.transform(X)
