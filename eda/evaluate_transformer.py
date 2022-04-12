# %%
import re
from collections import defaultdict
from itertools import product
from pathlib import Path
from pprint import pprint

import pandas as pd
from transformers import pipeline

data_path = Path("predatathon") / "data"
# RACISM_MODEL = "davidmasip/racism"

model_path = Path("models") / "artifacts" / "hf" / "v1_full"
racism_analysis_pipe = pipeline(
    "text-classification",
    model=str(model_path),
    tokenizer=str(model_path)
)


def clean_labels(results):
    for result in results:
        for score in result:
            if score["label"] == "LABEL_0":
                label = "Non-racist"
            else:
                label = "Racist"
            score["label"] = label


# %%
with (data_path / "racist_words_lemma.txt").open("r") as f:
    racist_words_lemma = f.read().splitlines()

racist_words_lemma.append("mena")
with (data_path / "racist_sentences.txt").open("r") as f:
    racist_sentences = f.read().splitlines()

racist_sentences_d = defaultdict(list)
# Create fake sentences
for word, sentence in product(racist_words_lemma, racist_sentences):
    racist_sentences_d[word].append(
        sentence.replace("<RACE>", f"los {word}s"))
    if word != "mena":
        racist_sentences_d[word].append(sentence.replace(
            "<RACE>", f"las {re.sub('o$', 'a', word)}s"))
    else:
        racist_sentences_d[word].append(
            sentence.replace("<RACE>", f"los {word}s"))


# %% Predict for each sentence
results = defaultdict(list)
for word, sentence in racist_sentences_d.items():
    results[word].append(racism_analysis_pipe(
        sentence, return_all_scores=True))


# %% Return racism score for each sentence
scores = defaultdict(list)
for word, score_list in results.items():
    for score in score_list[0]:
        scores[word].append(score[1]["score"])
# %% Average racist score
pd.DataFrame(scores).mean().sort_values()
pd.DataFrame(scores).median().sort_values()

# %%

