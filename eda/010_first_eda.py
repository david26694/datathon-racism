# %%
from pathlib import Path

import pandas as pd

data_path = Path("data")
input_path = data_path / "split"

data_path = Path("data")

pd.options.display.max_rows = 63

# %%
df = pd.read_csv(data_path / "labels_racism.csv", sep="|")

# %%
df.label.value_counts()

# %%
# with (data_path / "racist_data" / "potential_racist_words.txt").open("r") as f:
#     racist_words = f.read().splitlines()

# %%
racist_words = pd.read_csv(
    'predatathon/data/racist_words.txt', header=None)[0].to_list()

# %%
(
    df
    .assign(has_racist_word=lambda x: x.message.str.contains("|".join(racist_words)))
    .groupby(["label", "has_racist_word"], as_index=False)
    .size()
    .assign(
        n_rows=lambda x: x.groupby(
            ["label"])["size"].transform(lambda d: d.sum()),
        ratio=lambda x: x["size"] / x.n_rows
    )
)


# %%
racist_ratios = (
    df
    .groupby(["labeller_id", "label"], as_index=False)
    .size()
    # .rename(columns={"size": "n_rows"})
    .assign(
        n_rows=lambda x: x.groupby(["labeller_id"])[
            "size"].transform(lambda d: d.sum()),
        ratio=lambda x: x["size"] / x.n_rows
    )
    .query("label == 'racist'")
)

racist_ratios
# %%
df.groupby(["message"], as_index=False).size(
).sort_values("size", ascending=False)
# %%
repeated_messages = df.groupby(
    ["message"], as_index=False).size().query("size > 1").message.to_list()
len(repeated_messages)

# %%
len(df.message.unique())
# %%
df.loc[df.message.isin(repeated_messages)].sort_values(
    ["message", "labeller_id"])
# %%
many_labels = (
    df
    .loc[df.message.isin(repeated_messages)]
    .groupby("message", as_index=False)
    .agg(
        racists=('label', lambda x: (x == 'racist').sum()),
        non_racists=('label', lambda x: (x == 'non-racist').sum()),
        unknowns=('label', lambda x: (x == 'unknown').sum()),
    )
    .assign(
        some_racist=lambda x: x.racists > 0,
        some_non_racist=lambda x: x.non_racists > 0,
        some_unknown=lambda x: x.unknowns > 0,
    )
)
# %% STOP HERE
df
# %%
df.groupby("message", as_index=False).size().sort_values("size")

# %%
many_labels.groupby(["some_racist", "some_non_racist",
                    "some_unknown"], as_index=False).size()
# %%
many_labels.query("some_racist").query("some_non_racist").message.to_list()
# %%
many_labels.query("some_racist").query("some_non_racist")
# %%
df.groupby(["message", "labeller_id"]).filter(
    lambda x: len(x) > 1).sort_values(["message", "labeller_id"])

# %%


def labeller_bias(label_id):
    sentences_id = df.loc[df.message.isin(repeated_messages)].query(
        f"labeller_id == {label_id}").message.to_list()

    # Ratios of non-racist
    df_id = (
        df.loc[df.message.isin(sentences_id)]
        .assign(labeller_id=lambda x: x.labeller_id == label_id)
        .groupby(["labeller_id", "label"], as_index=False)
        .size()
        .assign(
            n_rows=lambda x: x.groupby(["labeller_id"])[
                "size"].transform(lambda d: d.sum()),
            ratio=lambda x: x["size"] / x.n_rows
        )
        .query("label == 'racist'")
    )

    others_racist_ratio = df_id.query("not labeller_id").ratio.values
    labeller_racist_ratio = df_id.query("labeller_id").ratio.values
    return float(labeller_racist_ratio - others_racist_ratio)


# %%
labellers = {}
df_bias = pd.DataFrame()
for i in range(1, 23):
    bias = labeller_bias(i)
    df_bias = df_bias.append(
        pd.DataFrame(
            {
                "labeller_id": [i],
                "bias": [bias],
                # "n_rows_shared": [n_rows_shared],
            }
        )
    )

df_bias = df_bias.reset_index(drop=True)
df_bias
# %%
rr = racist_ratios.merge(df_bias, on="labeller_id").assign(
    unbiased_ratio=lambda x: x.ratio - x.bias)

rr['abs_bias'] = abs(rr['bias'])
# %%
rr
# %%
(df.label == 'racist').astype(int)
# %%
train = pd.read_csv(data_path / "split" / "labels_racism_train.txt", sep="|")
test = pd.read_csv(data_path / "split" / "labels_racism_test.txt", sep="|")

# %%
(
    train
    .merge(rr.loc[:, ["labeller_id", "bias"]], how="left", on="labeller_id")
    .assign(label=lambda x: (x.label == 'racist').astype(int) - x.bias)
    .drop(columns=["bias"])
    .to_csv(data_path / "split" / "labels_racism_regression_train.txt", sep="|", index=False)
)
# %%
(
    test
    .merge(rr.loc[:, ["labeller_id", "bias"]], how="left", on="labeller_id")
    .assign(label=lambda x: (x.label == 'racist').astype(int) - x.bias)
    .drop(columns=["bias"])
    .to_csv(data_path / "split" / "labels_racism_regression_test.txt", sep="|", index=False)
)
# %%
