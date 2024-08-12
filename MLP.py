import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x
