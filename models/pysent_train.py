# %%
"""
Sklearn baseline

Careful: log_reg_baseline_validation.csv doesn't have unkowns
"""
import pickle
from pathlib import Path

import pandas as pd
from predatathon.utils.misc import ToListTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from stop_words import get_stop_words
from utils.classic_preprocessing import process_text
from utils.thresholds import predict_racism, threshold_optimisation

data_path = Path("data")
split_path = data_path / "split"
submission_path = data_path / "submission"
models_path = Path("models") / "artifacts"
models_path.mkdir(exist_ok=True, parents=True)
submission_path.mkdir(exist_ok=True, parents=True)

train_name = "log_reg_pysent"

# %%
train_df_raw = pd.read_csv(
    split_path / "labels_racism_train.txt", delimiter="|")
test_df_raw = pd.read_csv(split_path / "labels_racism_test.txt", delimiter="|")
eval_df_raw = pd.read_csv(data_path / "evaluation_public.csv", delimiter="|")

pysentimiento_feats = pd.read_csv(
    data_path / "features" / "pysentimiento_labels_racism.csv").drop(columns=["clean_message"])
train_df = train_df_raw.query("label != 'unknown'").merge(
    pysentimiento_feats, on=["message", "labeller_id", "label"])
test_df = test_df_raw.query("label != 'unknown'").merge(
    pysentimiento_feats, on=["message", "labeller_id", "label"])

full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# %% Spanish stop words
stop_words = get_stop_words('spanish')

# %% Apply preprocessing
train_df = process_text(train_df, stop_words)
test_df = process_text(test_df, stop_words)
full_df = process_text(full_df, stop_words)
eval_df = process_text(eval_df_raw, stop_words)

# %% Pipe creation
COLS2DROP = ["message", "labeller_id", "label", "processed_msg"]
cols2remain = set(train_df.columns).difference(COLS2DROP)


def to_dense(x):
    return x.todense()


pipe = Pipeline(
    [
        ("ml_features", FeatureUnion([
            ("grab_feats", ColumnTransformer([
                ("selector", "passthrough", list(cols2remain))
            ])),
            ("spacy_pipeline", Pipeline([
                ("grab_text", ColumnTransformer(
                    [("selector", "passthrough", ["processed_msg"])])),
                ("to_list", ToListTransformer()),
                ("count", CountVectorizer(max_features=200, ngram_range=(1, 2))),
                # Not needed if we don't scale
                ("denser", FunctionTransformer(to_dense, accept_sparse=True))
            ])),
        ])),
        # ("scaler", StandardScaler()),
        # Might as well use log reg
        ("model", LogisticRegression())
    ]
)

# %% Preprare train, test, train + test and eval sets
X = train_df.loc[:, cols2remain.union(["processed_msg"])]
y = train_df.label == 'racist'

X_test = test_df.loc[:, cols2remain.union(["processed_msg"])]
y_test = test_df.label == 'racist'

X_full = full_df.loc[:, cols2remain.union(["processed_msg"])]
y_full = full_df.label == 'racist'


# %% Create model and fit it
g = GridSearchCV(
    estimator=pipe,
    param_grid={
        # "count__max_features": [100, 200],
        # "count__ngram_range": [(1, 2)],
        "model__C": [0.01, 0.5, 1],
        # "model__threshold": np.linspace(0.4, 0.6, 10)
    },
    cv=5,
    scoring="f1"
)

g.fit(X, y)

# %% Best score, best params
print(g.best_score_)
print(g.best_params_)

# %% Threshold optimization
# use cross val predict to optimise threshold
preds = cross_val_predict(g.best_estimator_, X, y, method='predict_proba')
optimal_threshold = threshold_optimisation(preds, y)

# Use optimal threshold to predict train and test
train_preds = predict_racism(g.predict_proba(X), optimal_threshold)
test_preds = predict_racism(g.predict_proba(X_test), optimal_threshold)

# %%
optimal_threshold
# %% F1 score on test set -> 0.78
f1_score(y_test, (test_preds == 'racist').astype(int))

# %% F1 score on train set -> 0.80
f1_score(y, (train_preds == 'racist').astype(int))

# %% Save test predictions
test_df["racist_score"] = g.predict_proba(X_test)[:, 1]
test_df.drop(columns={"processed_msg"}).to_csv(
    submission_path / f"{train_name}_validation.csv", index=False)
# %% Retrain with all training data
g.fit(X_full, y_full)
preds = cross_val_predict(g.best_estimator_, X_full,
                          y_full, method='predict_proba')
optimal_threshold_full = threshold_optimisation(preds, y_full)

# %% Store g in models
pickle.dump(g, open(models_path / f"{train_name}.pkl", "wb"))
# Store threshold
with open(models_path / f"{train_name}_threshold.txt", "w") as f:
    f.write(str(optimal_threshold_full))
# %% Load models and submit
logreg = pickle.load(open(models_path / f"{train_name}.pkl", "rb"))
with open(models_path / f"{train_name}_threshold.txt", "r") as f:
    th_opt = float(f.read())

# %% Create submission file
predict_racism(logreg.predict_proba(X_test), th_opt)
