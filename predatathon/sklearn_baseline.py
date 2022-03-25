# %%
"""
Sklearn baseline
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words

from utils.classic_preprocessing import preprocess_line, preprocess_str

data_path = Path("data")


# %%
tweets_raw = pd.read_csv(data_path / "hate_speech_data.tsv", sep="\t")

# %%
stop_words = get_stop_words('spanish')

# %% Apply preprocessing
new_text = tweets_raw.text.apply(preprocess_line)
new_text = preprocess_str(new_text, stop_words)
tweets_raw["processed_text"] = new_text
# %% Word count
# new_text.str.split(expand=True).stack().value_counts().to_dict()

# %% Pipe creation
pipe = Pipeline(
    [
        ("count", CountVectorizer(max_features=20)),
        ("model", LogisticRegression())
    ]
)

# %%
X = tweets_raw.processed_text
y = tweets_raw.HS

# %%
g = GridSearchCV(
    estimator=pipe,
    param_grid={
        "count__max_features": [100, 200],
        "count__ngram_range": [(1, 2)],
        "model__C": [0.01, 0.5]
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
