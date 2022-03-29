from pathlib import Path

import pandas as pd

path = Path("data") / "submission" / "hf_v1_validation.csv"
# %%
df = pd.read_csv(path)
# %%
df.query("label == 'non-racist'").sort_values("racist_score",
                                              ascending=False).head(29).message.to_list()
# %%
df.query("label == 'racist'").sort_values(
    "racist_score").head(29).message.to_list()

# %%