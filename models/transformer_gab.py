#%%

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%

# %%

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

if Path("/kaggle").exists():
    input_path = Path("/kaggle") / "input" / "racism"
    output_path = Path("/kaggle") / "working"
    tmp_path = output_path / "tmp"
    models_path = output_path / "artifacts"
    models_path.mkdir(exist_ok=True, parents=True)
    tmp_path.mkdir(exist_ok=True, parents=True)
else:
    data_path = Path("data")
    input_path = data_path / "split"
    tmp_path = data_path / "tmp"
    tmp_path.mkdir(exist_ok=True, parents=True)
    models_path = Path("models") / "artifacts"
    output_path = models_path


train_df = pd.read_csv(input_path / "labels_racism_train.txt", delimiter="|")
test_df = pd.read_csv(input_path / "labels_racism_test.txt", delimiter="|")


def remove_weak(train_df):
    """remove inconsistent messages"""
    many_labels = (
        train_df
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

    many_labels['total_reviews'] = many_labels['racists'] + many_labels['non_racists'] + many_labels['unknowns']

    cut = many_labels.total_reviews.quantile(0.80)
    weak = many_labels.query('total_reviews > @cut & racists > 1 & non_racists > 1')
    train_df = train_df[~train_df.message.isin(weak['message'])]

    return train_df

train_df = remove_weak(train_df=train_df)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"


# %% Load tokenizer and train
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
)


def preprocess_function(examples):
    # txt = [x.replace("gitano", "negro") for x in txt]
    return tokenizer(examples["text"], truncation=True, padding=True)


# %% Load data


label_key = {'non-racist': 0, 'racist': 1, 'unknown': 2}

train_df["label"] = [label_key[item] for item in train_df.label]
test_df["label"] = [label_key[item] for item in test_df.label]

# exclude unknown
train_prep = train_df.query("label != 2").drop(columns='labeller_id', axis=1).rename(columns={
    'message': 'text'})

#
train_prep.to_csv(tmp_path / 'labels_racism_train.csv', index=False)
test_prep = test_df.query("label != 2").drop(columns='labeller_id', axis=1).rename(columns={
    'message': 'text'})
test_prep.to_csv(tmp_path / 'labels_racism_test.csv', index=False)

pd.concat([train_prep, test_prep]).to_csv(tmp_path / 'labels_racism_full.csv', index=False)

# %% Load ready for hf
dataset = load_dataset(path=str(tmp_path), data_files={
    'train': 'labels_racism_train.csv',
    'validation': 'labels_racism_test.csv',
    'full': 'labels_racism_full.csv',
    }
)


# %% Preprocess data
tokenized_data = dataset.map(preprocess_function, batched=True)


# %% Train model
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# %% Predict on validation
preds = trainer.predict(tokenized_data["validation"])

#%%

train_preds = trainer.predict(tokenized_data["train"])

#%%

def threshold_optimisation(preds, y, n_thresholds=200):

    df_thresholds = pd.DataFrame()
    for threshold in np.linspace(0, 1, n_thresholds):
        df_thresholds = df_thresholds.append(
            pd.DataFrame({
                "threshold": [threshold],
                "f1": [f1_score(preds[:, 1] > threshold, y)]
            })
        )

    return df_thresholds.sort_values("f1", ascending=False).head(1).threshold.to_list()[0]

#%%

y = train_preds.label_ids
y_preds = sigmoid(train_preds.predictions)
opt_threshold = threshold_optimisation(y_preds, y)
opt_threshold

#%%

y_valid = preds.label_ids
y_valid_preds = sigmoid(preds.predictions)
opt_threshold_valid = threshold_optimisation(y_valid_preds, y_valid)
opt_threshold_valid

#%%

print(f1_score(preds.label_ids, sigmoid(preds.predictions[:, 1]) > opt_threshold))

#%%

print(f1_score(train_preds.label_ids, sigmoid(train_preds.predictions[:, 1]) > opt_threshold))

#%%

trainer.save_model(models_path)
tokenizer.save_pretrained(models_path)


#%%

trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)


#%%

test_df_out = pd.read_csv(input_path / "labels_racism_test.txt", delimiter="|").query("label != 'unknown'")
test_df_out.assign(racist_score=sigmoid(preds.predictions[:, 1])).to_csv(output_path / "hf_v1_validation.csv", index=False)

#%%

# %% Create pipeline
p = pipeline(
    "text-classification", model=str(models_path), tokenizer=str(models_path))

# %% Save pipeline
p.save_pretrained(models_path / "pipe")

#%%

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["full"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

#%%

trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)


#%%


