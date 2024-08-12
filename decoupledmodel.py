import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from MLP import MLPClassifier
import utils.util as ut
import shap

def loadSingle(file, word_vectors, labels, Word_embedding_model):
    config = ut.readConfigs()
    with open(file, "r") as f:
        for line in f:
            word, label = line.strip().split(",")
            if config["CurrentEmbeddings"] == "ft":
                word_vector = Word_embedding_model[word]
            else:
                word_vector = Word_embedding_model.get_vector(word)
            word_vectors.append(word_vector)
            if config["CurrentLanguage"] == "zulu":
                labels.append(ut.convert(label))
            else:
                labels.append(ut.sepediConvert(label))

def training(Word_embedding_model):
    # Loading the vectors
    config = ut.readConfigs()
    word_vectors = []
    labels = []
    loadSingle(config["Training_Dataset"], word_vectors, labels, Word_embedding_model)   
    # Train test split and convert to tensors
    word_vectors = np.array(word_vectors)
    labels_encoded = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(word_vectors, labels_encoded, test_size=config["train_test_split"], stratify=labels_encoded)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    if config["CurrentEmbeddings"] == "ft":
        input_size = Word_embedding_model.get_dimension()
    else:    
        input_size = Word_embedding_model.vector_size
    num_classes = config["num_classes"]
    
    # Define parameters
    model = MLPClassifier(input_size, hidden_size1=config["hidden_size1"], output_size=num_classes, hidden_size2=config["hidden_size2"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Train Model
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print("Model Trained")
    torch.save(model.state_dict(), config["model_save_path"])
