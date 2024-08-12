import json
import torch.nn as nn
from MLP import MLPClassifier
from gensim.models.fasttext import FastTextKeyedVectors as gm
import fasttext
import torch
from torch.utils.data import DataLoader, TensorDataset
import utils.util as ut
import numpy as np
from decoupledmodel import training
from lime.lime_tabular import LimeTabularExplainer
import limewrapper as lw

class_names = ["Class 1", "Class 1a", "Class 2a", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9", "Class 10", "Class 11", "Class 12", "Class 13", "Class 14", "Class 15", "Class 16"]
array = np.arange(1, 19)

def loadwords(file, wordvectors, words, classes, convertedlabels, labels, word_embedding_model):
    config = ut.readConfigs()
    with open(file, "r") as f:
        for line in f:
            word, label = line.strip().split(",")
            if config["CurrentEmbeddings"] == "ft":
                word_vector = word_embedding_model[word]
                convertedlabels.append(ut.sepediConvert(label))
            else:
                word_vector = word_embedding_model.get_vector(word)
                convertedlabels.append(ut.convert(label))
            wordvectors.append(word_vector)
            words.append(word)
            classes.append(label)          
    return wordvectors, words, labels, convertedlabels

def Test(word_embedding_model): 
    # Set up the environment for Logging and saving results'
    config = ut.readConfigs()
    overallpath, summarypath = ut.testEnvironSetup(config["exNum"])
    print("Test Environment SetUp")
    unseenwordVecs = []
    convertedlabels = []
    words = []
    labels = []
    unseenwordVecs, words, labels, convertedlabels = loadwords(config["Testing_Dataset"], unseenwordVecs, words, labels, convertedlabels, labels, word_embedding_model)
    print("Words Loaded")
    
    # Load the unseen dataset
    unseenwordVecs = np.array(unseenwordVecs)
    unseen_dataset = TensorDataset(torch.from_numpy(unseenwordVecs).float())

    # Retrieve the input size of the model
    if config["CurrentEmbeddings"] == "ft":
        input_size = word_embedding_model.get_dimension()
    else:    
        input_size = word_embedding_model.vector_size

    # Create a dataloader for the unseen dataset and load the saved model
    unseen_loader = DataLoader(unseen_dataset, batch_size=config["batch_size"], shuffle=False)
    model = MLPClassifier(input_size, config["hidden_size1"], config["hidden_size2"], config["num_classes"])
    model.load_state_dict(torch.load(config["model_save_path"]))
    print("Model Loaded")
    # Set the model to evaluation mode and test the model
    model.eval()
    # Predict the labels of the words based on their vectors
    predicted_labels = []
    with torch.no_grad():
        for X_batch in unseen_loader:
            outputs = model(X_batch[0])
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.tolist())
    # Log the results of the test and the summary in the appropriate file
    ut.finalLog(summarypath, convertedlabels, predicted_labels, config["num_classes"], config["hidden_size1"], config["hidden_size2"], model, overallpath,labels, words)
    print("Test Complete")
    explainer = LimeTabularExplainer(unseenwordVecs, feature_names=None, class_names=array, mode='classification')
    wrapped_model = lw.ModelWrapper(model)
    # Explain the first 5 instances
    for i in range(5):
        exp = explainer.explain_instance(unseenwordVecs[i], wrapped_model.predict, num_features=10)
        exp.save_to_file(f'lime_explanation_{i}.html')

def automateTesting(numberOfTests,word_embedding_model):
    initial = config["exNum"]
    for i in range(initial,initial+numberOfTests):
        ut.updateConfigs("utils/config.json", "exNum", i)
        training(word_embedding_model)
        Test(word_embedding_model)

def AutoInputTesting(numberOfTests):
    for i in range(numberOfTests):
        config = ut.readConfigs()
        config["exNum"] = i
        temp_epochs = input("Enter the number of epochs: ")
        config["epochs"] = int(temp_epochs)
        temp_hidden_size1 = input("Enter the hidden size 1: ")
        config["hidden_size1"] = int(temp_hidden_size1)
        temp_learning_rate = input("Enter the learning rate: ")
        config["learning_rate"] = float(temp_learning_rate)
        temp_hidden_size2 = input("Enter the hidden size 2: ")
        config["hidden_size2"] = int(temp_hidden_size2)
        training(config["epochs"], config["hidden_size1"], config["learning_rate"], word_embedding_model, config["hidden_size2"])
        Test(word_embedding_model)

# Driver code to test the model
config = ut.readConfigs()
print("config file loaded")
# Load the word embeddings
if config["CurrentEmbeddings"] == "ft":
    word_embedding_model = fasttext.load_model(config["WordVecsSavePath"])
else:
    word_embedding_model = gm.load(config["WordVecsSavePath"])

print("Embeddings Loaded") #Signal to user
keyboard = input("Testing or Training:")
while True:
    if keyboard == "te":
        Test(word_embedding_model)
        keyboard = input("Testing or Training:")
    elif keyboard == "tr":
        tests = input("Enter the number of tests: ")
        automateTesting(int(tests),word_embedding_model)
        keyboard = input("Testing or Training:")
    else:
        quit()