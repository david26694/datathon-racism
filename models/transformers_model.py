# %%

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

data_path = Path("data")
split_path = data_path / "split"
models_path = Path("models") / "artifacts"
models_path.mkdir(exist_ok=True, parents=True)


train_df = pd.read_csv(split_path / "labels_racism_train.txt", delimiter="|")
test_df = pd.read_csv(split_path / "labels_racism_test.txt", delimiter="|")


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


train_df.query("label != 2").drop(columns='labeller_id', axis=1).rename(columns={
    'message': 'text'}).to_csv(split_path / 'labels_racism_train.csv', index=False)
test_df.query("label != 2").drop(columns='labeller_id', axis=1).rename(columns={
    'message': 'text'}).to_csv(split_path / 'labels_racism_test.csv', index=False)


# %% Load ready for hf
dataset = load_dataset(path=str(split_path), data_files={
    'train': 'labels_racism_train.csv',
    'validation': 'labels_racism_test.csv'
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

# data_collator = DataCollatorForTokenClassification(tokenizer)

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
print(f1_score(preds.label_ids, sigmoid(preds.predictions[:, 1]) > 0.25))

# %% Save model and tokenizer
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# %% Create pipeline
p = pipeline(
    "text-classification", model=str(model_path), tokenizer=str(model_path))

# %% Save pipeline
p.save_pretrained(model_path)
# %%
[x.replace("sirio", "moro") for x in dataset["train"]["text"]]
# %%
