import pickle
import argparse
import torch
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
    
class Experiment:

    def __init__(self, learning_rate, model, batch_size, dataset):
        self.model_name = model
        self.dataset_path = dataset
        self.learning_rate = learning_rate
        #self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.extract_dataset(dataset)
        #dataset = load_dataset("custom dataset")
        #tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        #small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        #small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

        #model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)

        #metric = evaluate.load("accuracy")

        #training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    def extract_dataset(self, dataset):
        #convert dataset into json for dataset loader
        dataset_file = open(dataset, 'rb')
        data = pickle.load(dataset_file)
        #THE STRUCTURE OF THE DATA IS TOO RIGID. NEED TO MAKE IT MORE FLEXIBLE!
        print(data.values.tolist()[0][6], "[SEP]", data.values.tolist()[0][13])
        #dataset = Dataset.from_pandas(data)
        #dataset = Dataset.from_list(my_list)
        #print(dataset)

    #def tokenize_function(examples):
        #return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    #def train_and_eval(self):

        #trainer = Trainer(
            #model=model,
            #args=training_args,
            #train_dataset=train_dataset,
            #eval_dataset=eval_dataset,
            #compute_metrics=compute_metrics,
        #)

        #trainer.train()

    #def compute_metrics(eval_pred):
        #logits, labels = eval_pred
        #predictions = np.argmax(logits, axis=-1)
        #return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="next_equation_selection.pkl", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
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
    experiment = Experiment(learning_rate = args.lr, batch_size = args.batch_size, model = args.model, dataset = data_path)
    #experiment.train_and_eval()
                

