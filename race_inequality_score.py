# %%
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd
from transformers import pipeline

data_path = Path("data")
RACISM_MODEL = "davidmasip/racism"
racism_analysis_pipe = pipeline("text-classification",
                                model=RACISM_MODEL, tokenizer=RACISM_MODEL)


# %%
with (data_path / "racist_words_lemma.txt").open("r") as f:
    racist_words_lemma = f.read().splitlines()

with (data_path / "racist_sentences.txt").open("r") as f:
    racist_sentences = f.read().splitlines()

racist_sentences_d = defaultdict(list)
# Create fake sentences
for word, sentence in product(racist_words_lemma, racist_sentences):
    racist_sentences_d[word].append(
        sentence.replace("<RACE>", f"los {word}s"))
    racist_sentences_d[word].append(sentence.replace(
        "<RACE>", f"las {re.sub('o$', 'a', word)}s"))

# %% Predict for each sentence
results = defaultdict(list)
for word, sentence in racist_sentences_d.items():
    results[word].append(racism_analysis_pipe(
        sentence, return_all_scores=True))


# %% Return racism score for each sentence
scores = defaultdict(list)
for word, score_list in results.items():
    for score in score_list[0]:
        scores[word].append(score[0]["score"])
# %% Average racist score
pd.DataFrame(scores).mean()

# %%
