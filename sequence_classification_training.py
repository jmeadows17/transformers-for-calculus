import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, dataset_path):
        self.model_name = model
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.dataset = self.process_dataset(dataset_path, neg)
        self.tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
        self.metric = evaluate.load("glue", "mrpc")
        self.training_args = TrainingArguments(
                output_dir="output/"+self.model_name,
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
            context = (str(example['derivation'][:-1]) + ' [SEP] ' + str(example['derivation'][-1][0])).replace('[[','[').replace(']]',']').replace('\\\\','\\').replace("'","").replace("\\,","")
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
        dataset_split = dataset.train_test_split(test_size=0.2)
        return dataset_split

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", max_length = self.max_length, truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        majority_class_preds = [1 for pred in predictions]
        majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=predictions, references=labels)
        return score

    def train_and_eval(self):
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.tokenized_datasets["train"],
            eval_dataset = self.tokenized_datasets["test"],
            compute_metrics = self.compute_metrics
        )
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="next_equation_selection_steps=2.json", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="bert-base-uncased", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=256, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=float, default=3.0, nargs="?",
                    help="Num epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--neg", type=int, default=1, nargs="?",
                    help="Max number of negative examples")

    args = parser.parse_args()
    dataset = args.dataset
    data_path = "data/"+dataset
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(
            learning_rate = args.lr, 
            batch_size = args.batch_size, 
            neg = args.neg,
            max_length = args.max_length,
            epochs = args.epochs, 
            model = args.model, 
            dataset_path = data_path
            )
    experiment.train_and_eval()
                

