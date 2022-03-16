# %%
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

data_path = Path("data")
hf_data = Path("hf_data")
model_path = Path("racism")
data_path.mkdir(exist_ok=True)
hf_data.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)


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
    return tokenizer(examples["text"], truncation=True)


# %% Load data
try:
    tweets_raw = pd.read_csv(
        data_path / "hate_speech_data.tsv",
        sep="\t").rename(columns={"HS": "label"})
except FileNotFoundError:
    tweets_raw = pd.read_csv(
        "https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-Spanish-A/train_dev_es_merged.tsv", sep="\t").rename(columns={"HS": "label"})

    # Split and save (prepare for HF)
    from sklearn.model_selection import train_test_split
    train, validation = train_test_split(
        tweets_raw.loc[:, ["text", "label"]], random_state=42)
    train.to_csv(hf_data / "train.csv", index=False)
    validation.to_csv(hf_data / "validation.csv", index=False)

# %% Load ready for hf
dataset = load_dataset(
    'csv', data_files={
        'train': str(hf_data / "train.csv"),
        'validation': str(hf_data / "validation.csv")
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
    num_train_epochs=1,
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


# %% Print roc
print(roc_auc_score(preds.label_ids, sigmoid(preds.predictions[:, 1])))

# %% Save model and tokenizer
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# %% Create pipeline
p = pipeline(
    "text-classification", model=str(model_path), tokenizer=str(model_path))

# %% Save pipeline
p.save_pretrained(model_path)
# %%
