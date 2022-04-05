# %%
from pathlib import Path
import pandas as pd
import os
os.getcwd()

data_path = Path("data")
input_path = data_path / "split"

data_path = Path("data")

pd.options.display.max_rows = 63

# %%
df = pd.read_csv(data_path / "labels_racism.csv", sep="|")

with (data_path /"racist_data/racist_words_lemma.txt").open("r") as f:
    racist_words_lemma = f.read().splitlines()


dff = pd.DataFrame([])

for i in racist_words_lemma:
    df_counts = pd.DataFrame({"word":[i],
                  "count":sum(df['message'].apply(lambda x: x.count(i)))})
    df_counts.set_index('word')
    dff = dff.append(df_counts)



dff.set_index('word')
