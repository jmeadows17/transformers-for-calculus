import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
    
class Experiment:

    def __init__(self, learning_rate, model, batch_size, dataset_path):
        self.model_name = model
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.dataset = self.process_dataset(dataset_path)
        self.tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=1)
        self.metric = evaluate.load("f1")
        self.training_args = TrainingArguments(output_dir="output", evaluation_strategy="epoch", num_train_epochs = 3.0, learning_rate = 5e-5, per_device_train_batch_size = 8)

    def process_dataset(self, dataset_path):
        #convert dataset into json for dataset loader
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        formatted_examples = []
        for example in tqdm(d_json, desc="Loading Dataset"):
            #create an entry for each positive example
            positive_ids = list(np.array(example["positive_idxs"]) - 1)
            candidate = [example["derivation"][equation_id][0] for equation_id in positive_ids]
            context = [example["derivation"][equation_id][0] for equation_id in list(set(range(0,len(example["derivation"]))).difference(set(positive_ids)))]
            input_text = " ".join(context) + " [SEP] " + " ".join(candidate)
            formatted_examples.append({"text": input_text, "label": 1.0})
            #create an entry for each negative example
            for negative in example["negatives"]:
                input_text = " ".join(context) + " [SEP] " + negative
                formatted_examples.append({"text": input_text, 'label': 0.0})
        #split randomly between train, dev, and test set
        dataset = Dataset.from_list(formatted_examples)
        dataset_split = dataset.train_test_split(test_size=0.2)
        return dataset_split

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train_and_eval(self):

        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.tokenized_datasets["train"],
            eval_dataset = self.tokenized_datasets["test"],
            compute_metrics = self.compute_metrics,
        )

        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="next_equation_selection.json", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="bert-base-uncased", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")

    args = parser.parse_args()
    dataset = args.dataset
    data_path = "data/"+dataset
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(learning_rate = args.lr, batch_size = args.batch_size, model = args.model, dataset_path = data_path)
    experiment.train_and_eval()
                

