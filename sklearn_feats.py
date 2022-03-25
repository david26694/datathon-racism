# %%
"""
Sklearn baseline
"""
from pathlib import Path

import detoxify
import pandas as pd
from sklearn.compose import (ColumnTransformer, make_column_selector,
                             make_column_transformer)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from stop_words import get_stop_words

from utils.classic_preprocessing import preprocess_line, preprocess_str
from utils.misc import ToListTransformer

data_path = Path("data")


# %%
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")
bullying_feats = pd.read_csv(data_path / "bullying_scores.csv")
dehate_feats = pd.read_csv(data_path / "dehatebert_feats.csv")
xlm_feats = pd.read_csv(data_path / "xlm_roberta_feats.csv")
detoxify = pd.read_csv(data_path / "detoxify_feats.csv")
pysentimiento_feats = pd.read_csv(data_path / "pysentimiento_feats.csv")
tweets = (
    tweets_raw
    .merge(bullying_feats, on="id", how="left")
    .merge(dehate_feats, on="id", how="left")
    .merge(xlm_feats, on="id", how="left")
    .merge(detoxify, on="id", how="left")
    # .merge(pysentimiento_feats, on="id", how="left")
)

# %%
stop_words = get_stop_words('spanish')

# %% Apply preprocessing
new_text = tweets.text.apply(preprocess_line)
new_text = preprocess_str(new_text, stop_words)
tweets["processed_text"] = new_text


# %% Pipe creation
COLS2DROP = ["id", "text", "processed_text", "HS", "TR", "AG"]
cols2remain = set(tweets.columns).difference(COLS2DROP)
pipe = Pipeline(
    [
        ("ml_features", FeatureUnion([
            ("grab_feats", ColumnTransformer([
                ("selector", "passthrough", list(cols2remain))
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
        # Might as well use log reg
        ("model", LogisticRegression())
    ]
)

# %%
X = tweets.drop(columns=set(COLS2DROP).difference(["processed_text"]))
y = tweets.HS

X
# %%
g = GridSearchCV(
    # estimator=Pipeline([("model", LogisticRegression())]),
    estimator=pipe,
    param_grid={
        # "model__C": [0.1, 0.5, 1.0],
        # # Best l1 ratio is 0
        # "model__l1_ratio": [0.0]
        # # "model__l1_ratio": [0.0, 0.5, 1.0]
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
