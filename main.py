import pandas as pd
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tokenization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Helper function: Balanced subset
def get_balanced_subset(dataset_split, num_per_class):
    pos_samples = dataset_split.filter(lambda x: x["label"] == 1).select(range(num_per_class))
    neg_samples = dataset_split.filter(lambda x: x["label"] == 0).select(range(num_per_class))
    return concatenate_datasets([pos_samples, neg_samples]).shuffle(seed=42)

# Load IMDB and balance it
dataset = load_dataset("imdb")
train_data = get_balanced_subset(dataset["train"], 2000)  # 2000 pos + 2000 neg = 4000
val_data = get_balanced_subset(dataset["test"], 500)      # 500 pos + 500 neg = 1000

# Tokenize
tokenized_train = tokenizer(list(train_data["text"]), padding=True, truncation=True)
tokenized_val = tokenizer(list(val_data["text"]), padding=True, truncation=True)


# Custom Dataset
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(tokenized_train, train_data["label"])
val_dataset = IMDbDataset(tokenized_val, val_data["label"])
from collections import Counter

print("Train label distribution:", Counter(train_dataset.labels))
print("Validation label distribution:", Counter(val_dataset.labels))


# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load model with proper initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Initialize weights properly - this is important for classification
model.classifier.weight.data.normal_(mean=0.0, std=0.02)
model.classifier.bias.data.zero_()

model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print("Starting training...")
trainer.train()

# Evaluate the model
print("\nEvaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save model and tokenizer
model_path = './model/'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")