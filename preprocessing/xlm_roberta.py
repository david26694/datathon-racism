# %%
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

data_path = Path("data")


# MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"
MODEL_NAME = "daveni/twitter-xlm-roberta-emotion-es"

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def batch_predictor(model, tokenizer, text, batch_size=32):
    logit_list = []

    for i in tqdm(range(0, len(text), batch_size)):
        max_idx = min(i + batch_size, len(text))
        encoded_input = tokenizer(
            text[i:max_idx],
            truncation=True,
            return_tensors="pt",
            padding=True
        )
        output = model(**encoded_input)
        logits = (output.logits.detach().numpy())
        logit_list.append(logits)

    return logit_list


# %%
tweets_raw = pd.read_csv(
    data_path / "hate_speech_data.tsv", sep="\t")
text = (
    tweets_raw.text
    # Maximum length allowed is 512 tokens, but they are not exaclty words
    .apply(lambda x: x.split()[:100])
    .apply(lambda x: " ".join(x))
    .to_list()
)


# %% Time to preprocess the text
time_start = time.time()
batch_size = 32
batch_predictor(model=model, tokenizer=tokenizer,
                text=text, batch_size=batch_size)
time_end = time.time()

# %%
print(time_end - time_start)
# %%
logits = pd.concat([pd.DataFrame(logit)
                   for logit in logit_list]).reset_index(drop=True)
# %%
n_cols = logits.shape[1]
feature_names = [f"emotion_{i}" for i in range(n_cols)]
logits.columns = feature_names
logits.assign(id=tweets_raw.id).to_csv(
    data_path / "xlm_roberta_feats.csv", index=False)
# %%
