# %%
"""
Sklearn baseline
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import (ColumnTransformer, make_column_selector,
                             make_column_transformer)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from stop_words import get_stop_words
from config import PYSENTIMIENTO_FEATS

from utils.classic_preprocessing import preprocess_line, preprocess_str
from utils.misc import ToListTransformer

data_path = Path("data")


# %%
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")
spacy_feats = pd.read_csv(data_path / "spacy_feats.csv")
bullying_feats = pd.read_csv(data_path / "bullying_scores.csv")
tweets = (
    tweets_raw
    .merge(spacy_feats, on="id", how="left")
    .merge(bullying_feats, on="id", how="left")
)

# %%
stop_words = get_stop_words('spanish')

# %% Apply preprocessing
new_text = tweets.text.apply(preprocess_line)
new_text = preprocess_str(new_text, stop_words)
tweets["processed_text"] = new_text


# %% Pipe creation
pipe = Pipeline(
    [
        ("ml_features", FeatureUnion([
            ("grab_text", make_column_transformer(
                (FunctionTransformer(), make_column_selector(pattern="spacy|bullying")))),
            ("grab_pysentimiento", ColumnTransformer([
                ("selector", "passthrough", PYSENTIMIENTO_FEATS)
            ])),
            ("spacy_pipeline", Pipeline([
                ("grab_text", ColumnTransformer(
                    [("selector", "passthrough", ["processed_text"])])),
                ("to_list", ToListTransformer()),
                ("count", CountVectorizer(max_features=200, ngram_range=(1, 2))),
                # Not needed if we don't scale
                # ("denser", FunctionTransformer(
                #     lambda x: x.todense(), accept_sparse=True))
            ])),
        ])),
        # ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]
)

# %%
X = tweets
y = tweets.HS

# %%
g = GridSearchCV(
    estimator=pipe,
    param_grid={
        "model__C": [0.1, 0.5, 1.0],
        # Best l1 ratio is 0
        "model__l1_ratio": [0.0]
        # "model__l1_ratio": [0.0, 0.5, 1.0]
    },
    cv=5,
    scoring="average_precision"
)

g.fit(X, y)

# %%
print(g.best_score_)
# %%
print(g.best_params_)

# %%
