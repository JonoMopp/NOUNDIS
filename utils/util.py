import array
from datetime import datetime as dt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import load_metric
import json
import os
import torch

config_path = os.path.join(os.getcwd(),'utils','config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

def logmessage(currentPath,message):
    with open(currentPath, "a") as f:
        f.write(message+"\n")

def createfile(filename):
    with open(filename, "w") as f:
        f.write("Log file created at: " + filename+"\n")

def convert(zulu):
    if zulu == "1a":
        return 2
    elif zulu == "2a":
        return 3
    elif zulu == "1":
        return int(zulu)
    else: 
        return int(zulu)+2

def revert(zulu):
    if zulu == 2:
        return "1a"
    elif zulu == 3:
        return "2a"
    elif zulu == 1:
        return "1"
    else: 
        return str(zulu-2)

def testEnvironSetup(exNum):
        # Get the current date
        current_date = dt.today()
        date_string = current_date.strftime("%Y-%m-%d")
        # Get the current language from config
        current = config["CurrentLanguage"]
        # Create the directory structure
        logPath = os.path.join("Logs", current, date_string)
        exNumPath = os.path.join(logPath, f"Experiment{exNum}")
        os.makedirs(logPath, exist_ok=True)
        print(f"Directory created: {logPath}")
        # Create experiment directory
        os.makedirs(exNumPath, exist_ok=True)
        print(f"Directory created: {exNumPath}")
        # Create files
        overallPath = os.path.join(exNumPath, "Overall.txt")
        summaryPath = os.path.join(exNumPath, "Summary.txt")
        createfile(overallPath)
        createfile(summaryPath)
        return overallPath, summaryPath

def scores(labels,predicted):
    accuracy = accuracy_score(labels,predicted)
    precision = precision_score(labels,predicted,average='weighted')
    recall = recall_score(labels,predicted,average='weighted')
    f1 = f1_score(labels,predicted,average='weighted')
    return accuracy, precision, recall, f1



def compute_metrics(p):
    metric = load_metric("accuracy")
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return metric.comp(predictions=predictions, references=labels)

def readConfigs():
    with open('utils/config.json', 'r') as f:
        config = json.load(f)
    return config

def finalLog(summarypath, convertedlabels, predicted_labels, num_classes, hidden_size, hiddenSize2, model, overallpath,labels, words):
    config = readConfigs()
    arr = array.array('i', [0] * num_classes)
    count = 0
    correct = 0
    incorrect = 0
    for word, label in zip(words, predicted_labels):
        t = ""
        if config["CurrentLanguage"] == "zulu":
            t = convert(labels[count])
            logmessage(overallpath, f"{word}, PC:{revert(label)}, TC:{revert(t)}")
        else:
            t = sepediConvert(labels[count])
            logmessage(overallpath, f"{word}, PC:{sepediRevert(label)}), TC:{sepediRevert(t)}")    
        if label == t:
            correct += 1
        else: 
            arr[int(t)] += 1
            incorrect += 1
        count += 1
    logmessage(summarypath, f"Summary for MLP with epoch:{config['epochs']}, hidden_size:{hidden_size}, hidden_size2: {hiddenSize2}, lr:{config['learning_rate']}, Vectors from: {config['VectorTypes']}")
    logmessage(summarypath, f"Correct: {correct}")
    logmessage(summarypath, f"Incorrect: {incorrect}")
    accuracy, precision, recall, f1 = scores(convertedlabels, predicted_labels)
    logmessage(summarypath, f"Accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")
    if accuracy > config["save_threshold"]:
        torch.save(model.state_dict(), f'{config["HighAccuracyModelPath"]}{accuracy},{config["epochs"]}.pth')
        print(f"Model accuracy: {accuracy}. Epoch: {config['epochs']}")
    logmessage(summarypath, f"Precision: {precision}")
    logmessage(summarypath, f"Recall: {recall}")
    logmessage(summarypath, f"F1: {f1}")
    logmessage(summarypath, "Incorrect Totals by class:")
    for i in range(num_classes):
        if i == 0:
            logmessage(summarypath, f"Class 1: {arr[i]}")
        elif i == 1:
            logmessage(summarypath, f"Class 1a: {arr[i]}")  
        elif i == 2:
            logmessage(summarypath, f"Class 2a: {arr[i]}")
        else:  
            logmessage(summarypath, f"Class {i-1}: {arr[i]}")

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def updateConfigs(file_path,Var2Change,newValue):
    with open(file_path, 'r') as f:
        config = json.load(f)
    config[Var2Change] = newValue
    write_json(file_path,config)
    print("Experiment updated")

def sepediConvert(sep):
    if  sep == "1a":
        return 2
    elif sep == "2b":
        return 3
    elif sep == "1":
        return int(sep)
    else: 
        return int(sep)+2
    
def sepediRevert(sep):
    if sep == 2:
        return "1a"
    elif sep == 3:
        return "2b"
    elif sep == 1:
        return "1"
    else: 
        return str(sep-2)
    
