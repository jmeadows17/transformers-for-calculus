import json
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"    

class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, neg, dataset_path):
        self.model_name = model
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model)# MANUALLY ENTER THE TOKENIZER NAME
        self.dataset = self.process_dataset(dataset_path, neg)
        self.tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2).to(device)
        self.metric = evaluate.load("glue", "mrpc")
        self.training_args = TrainingArguments(
                output_dir= os.getcwd() + "\\output",
                logging_steps = 500,
                evaluation_strategy="steps",
                eval_steps = 500,
                num_train_epochs = epochs,
                learning_rate = learning_rate,
                per_device_train_batch_size = batch_size
                )

    def process_dataset(self, dataset_path, neg):
        #convert dataset into json for dataset loader
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        formatted_examples = []
        for example in tqdm(d_json, desc="Loading Dataset"):
            #create an entry for each positive example
            positive_ids = list(np.array(example["positive_idxs"]) - 1)
            candidate = [example["derivation"][equation_id][1] for equation_id in positive_ids]
            context = (str(example['derivation'][:-1]) + ' [SEP] ' + str(example['derivation'][-1][0])).replace('[[','[').replace(']]',']').replace('\\\\','\\')
            input_text = context + " [SEP] " + " ".join(candidate)
            formatted_examples.append({"text": input_text, "label": 1})
            #create an entry for each negative example
            count_neg = 0
            for negative in example["negatives"]:
                if count_neg == neg:
                    break
                input_text = context + " [SEP] " + negative
                formatted_examples.append({"text": input_text, 'label': 0})
                count_neg += 1
        print("Data examples", formatted_examples[:4])
        #split randomly between train, dev, and test set
        dataset = Dataset.from_list(formatted_examples)
        dataset_split = dataset.train_test_split(test_size=0.99)
        return dataset_split

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        majority_class_preds = [1 for pred in predictions]
        majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=predictions, references=labels)
        print(score)
        return score

    def train_and_eval(self):
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.tokenized_datasets["train"],
            eval_dataset = self.tokenized_datasets["test"],
            compute_metrics = self.compute_metrics
        )
        trainer.evaluate()

if __name__ == '__main__':
    
    model = os.getcwd() + "\\models\\NES_steps=2.json_roberta-base"
    data_path = os.getcwd() + "\\data\\EVAL_NES_steps=2.json"
    
    
    #data_path = os.getcwd() + "\\data\\EVAL_NES_steps=3.json"
    #data_path = os.getcwd() + "\\data\\NES_VAR_RE_steps=3.json"
    #data_path = os.getcwd() + "\\data\\NES_EXPR_EXC_steps=3.json"
    #data_path = os.getcwd() + "\\data\\NES_OP_SWAP_steps=3.json"
    
    #data_path = os.getcwd() + "\\data\\EVAL_NES_steps=4.json"
    #data_path = os.getcwd() + "\\data\\NES_VAR_RE_steps=4.json"
    #data_path = os.getcwd() + "\\data\\NES_EXPR_EXC_steps=4.json"
    #data_path = os.getcwd() + "\\data\\NES_OP_SWAP_steps=4.json"
    
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(
            learning_rate = 5e-5,
            batch_size = 8, 
            neg = 1, 
            epochs = 1, 
            model = model, 
            dataset_path = data_path
            )
    experiment.train_and_eval()