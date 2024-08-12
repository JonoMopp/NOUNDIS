from datasets import Dataset
import os
import utils.util as ut

def createDataSet(filepath):
    with open(filepath, "r") as f:
        words = []
        labels = []
        for line in f:
            word, label = line.strip().split(",")
            words.append(word)
            if ut.readConfigs()["CurrentLanguage"] == "zulu":
                labels.append(ut.convert(label))
            else:
                labels.append(ut.sepediConvert(label))
    return words, labels

def print_samples(tokenized_datasets, num_samples=5):
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        for key, value in tokenized_datasets[i].items():
            print(f"  {key}: {value}")
        print()

def createLog(exnum):
    with open("results/Serengeti.txt", "w") as f:
        f.write("Epoch number: " + str(exnum) + "\n") 

def logmsg(msg):
    with open("results/Serengeti.txt", "a") as f:
        for i in range(len(msg)):
            f.write(msg[i] + "\n")
