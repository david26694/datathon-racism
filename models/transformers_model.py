# %%

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, pipeline, \
    TrainingArguments, Trainer, DataCollatorForTokenClassification
from pathlib import Path

"""
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)
"""

data_path = Path("data")
hf_data = Path("hf_data")
model_path = Path("models") / "artifacts"
data_path.mkdir(exist_ok=True)
hf_data.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)

data_path = Path("data")
split_path = data_path / "split"
submission_path = data_path / "submission"
models_path = Path("models") / "artifacts"
models_path.mkdir(exist_ok=True, parents=True)
submission_path.mkdir(exist_ok=True, parents=True)


train_df = pd.read_csv(split_path / "labels_racism_train.txt", delimiter="|")
test_df = pd.read_csv(split_path / "labels_racism_test.txt", delimiter="|")
eval_df = pd.read_csv(data_path / "evaluation_public.csv", delimiter="|")




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
    return tokenizer(examples["text"], truncation=True)


# %% Load data


train_df.drop(columns='labeller_id', axis=1).rename(columns ={'message':'text'}).to_csv('data/split/labels_racism_train.csv', index = False)
test_df.drop(columns='labeller_id', axis=1).rename(columns ={'message':'text'}).to_csv('data/split/labels_racism_test.csv', index = False)

# %% Load ready for hf
dataset = load_dataset(path = 'data/split', data_files={
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

data_collator = DataCollatorForTokenClassification(tokenizer)

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
[x.replace("sirio", "moro") for x in dataset["train"]["text"]]
# %%
