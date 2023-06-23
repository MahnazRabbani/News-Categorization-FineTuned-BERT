import os
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def load_data_and_model(dataset_name, model_name, num_labels):
    raw_datasets = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return raw_datasets, tokenizer, model

def tokenize_data(raw_datasets, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    
    return small_train_dataset, small_eval_dataset, full_train_dataset, full_eval_dataset

def get_training_args(path, evaluation_strategy):
    return TrainingArguments(path, evaluation_strategy=evaluation_strategy)

def get_metric(metric_name):
    return load_metric(metric_name)

def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_trainer(model, args, train_dataset, eval_dataset, compute_metrics):
    return Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

def evaluate_model(trainer):
    return trainer.evaluate()

def save_model(trainer, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    trainer.save_model(model_path)

def main():
    try:
        raw_datasets, tokenizer, model = load_data_and_model("ag_news", "bert-base-uncased", 4)
        small_train_dataset, small_eval_dataset, full_train_dataset, full_eval_dataset = tokenize_data(raw_datasets, tokenizer)
        training_args = get_training_args("test_trainer", "epoch")
        metric = get_metric("accuracy")

        trainer = get_trainer(model, training_args, small_train_dataset, small_eval_dataset, lambda eval_pred: compute_metrics(eval_pred, metric))
        eval_results = evaluate_model(trainer)
        print(eval_results)

        save_model(trainer, './models')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
