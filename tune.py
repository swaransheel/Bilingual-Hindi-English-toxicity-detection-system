#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import evaluate  # Updated import
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine‑tune a text classifier on successful_urls.csv for toxicity"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to successful_urls.csv"
    )
    parser.add_argument(
        "--model", type=str, default="distilbert-base-uncased",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./toxicity_classifier",
        help="Where to save fine‑tuned model"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Per-device batch size"
    )
    return parser.parse_args()

def load_and_prepare_data(csv_path):
    # 1) Load into pandas
    df = pd.read_csv(csv_path)

    # 2) Keep only definite labels
    df = df[df['contains_toxicity'].isin(['Yes','No'])].reset_index(drop=True)

    # 3) Map to binary label
    df['label'] = df['contains_toxicity'].map({'No': 0, 'Yes': 1})

    # 4) Split by partition
    train_df = df[df['partition']=='train']
    eval_df  = df[df['partition'].isin(['dev','devtest'])]

    # 5) Convert to Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df)
    eval_ds  = Dataset.from_pandas(eval_df)

    return train_ds, eval_ds

def tokenize_datasets(train_ds, eval_ds, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(batch['audio_file_transcript'], truncation=True)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds  = eval_ds.map(tokenize_fn,  batched=True)
    return train_ds, eval_ds

def main():
    args = parse_args()

    # Load & prepare
    train_ds, eval_ds = load_and_prepare_data(args.csv)

    # Tokenizer & collator
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds, eval_ds = tokenize_datasets(train_ds, eval_ds, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )

    # Metric - Updated to use evaluate library
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        logging_steps=50,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[✓] Best model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
