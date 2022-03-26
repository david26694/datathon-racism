# %%
from pathlib import Path

import numpy as np
import pandas as pd

data_path = Path("data")
data_path.mkdir(exist_ok=True)
split_path = data_path / "split"
split_path.mkdir(exist_ok=True)

# %%
df = pd.read_csv(data_path / "labels_racism.csv", sep="|")

# %%
messages = df.message.unique()

# %%
np.random.seed(42)
messages_train = np.random.choice(
    messages, size=int(len(messages) * 0.6), replace=False)
# %%
messages_train
# %%
df_train = df.loc[df.message.isin(messages_train)].reset_index(drop=True)
df_test = df.loc[~df.message.isin(messages_train)].reset_index(drop=True)

# %%
df_train
# %%
len(df_test) / len(df_train)
# %% Get proportion of labels in df_train
df_train.label.value_counts() / len(df_train)
# %%
df_test.label.value_counts() / len(df_test)

# %%
df.groupby(["message", "labeller_id"]).filter(
    lambda x: len(x) > 1).sort_values(["message", "labeller_id"])

# %%
df_train.to_csv(split_path / "labels_racism_train.txt", sep="|", index=False)
df_test.to_csv(split_path / "labels_racism_test.txt", sep="|", index=False)
# %%
