# %%
from pathlib import Path

import pandas as pd

data_path = Path("data")
input_path = data_path / "split"

data_path = Path("data")

pd.options.display.max_rows = 63

# %%
df = pd.read_csv(data_path / "labels_racism.csv", sep="|")
df

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
