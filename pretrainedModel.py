from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import utils.pretrainedUtils as pretrained
import utils.util as ut
import numpy as np
from datasets import load_metric, DatasetDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as Dataset
import torch

config = ut.readConfigs()

class AfricanLanguageDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)
    


def PretrainedModelTraining(epochs):
    pretrained.createLog(epochs)
    words, labels = pretrained.createDataSet(config["Training_Dataset"]) # Creates HuggingFace Dataset
    Testwords, Testlabels = pretrained.createDataSet(config["Testing_Dataset"]) # Creates HuggingFace Dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(words, labels, test_size=.2)
    tokenizer = AutoTokenizer.from_pretrained(config["Serengeti"]) # Loads tokenizer
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(Testwords, truncation=True, padding=True)
    train_dataset = AfricanLanguageDataset(train_encodings, train_labels)
    val_dataset = AfricanLanguageDataset(val_encodings, val_labels)
    test_dataset = AfricanLanguageDataset(test_encodings, Testlabels)

    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=epochs,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=100,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True 
    )

    model = AutoModelForSequenceClassification.from_pretrained(config["Serengeti"], num_labels=30)
    trainer = Trainer(
        model=model,                        
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),            
    )

    trainer.train()
    results = trainer.evaluate(test_dataset)
    print(results)
    metric = load_metric("accuracy")
    predictions, labels, metrics = trainer.predict(test_dataset)
    accuracy = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
    # Prepare metrics for testing
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")
    precision = precision_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="weighted")
    recall = recall_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="weighted")
    scores = [accuracy,precision, recall, f1]
    pretrained.logmsg(scores)

def PretrainedModelTesting(Repeats,epochs):
    for i in range(epochs,epochs+Repeats):
        PretrainedModelTraining(i)

epochs = int(input("Enter the number of epochs: "))
repeats = int(input("Enter the number of repeats: "))
while True:
    PretrainedModelTesting(epochs,repeats) # Enter the number of epochs and the number of times to perform the experiment
    epochs = int(input("Enter the number of epochs: "))
    repeats = int(input("Enter the number of repeats: "))
    if epochs == "":
        break

