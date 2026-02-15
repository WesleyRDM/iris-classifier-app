import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def take_dataset():
    
    data = load_iris()
    X = data['data']
    y = data['target']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
               
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    return X,y

def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class PML(nn.Module):
    def __init__(self,in_feature,hidden_layer,out_feature):
        super().__init__()
        self.linear1 = nn.Linear(in_feature, hidden_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer,out_feature)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

X, y = take_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)

in_feature = X_train.shape[1]
hidden_layer=10
out_feature=3
learning_rate=0.05
epoch = 100


model = PML(in_feature, hidden_layer, out_feature)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#======== Training =========:

for i in range(epoch):
    optimizer.zero_grad()
    outputs = model(X_train)

    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)        
    correct = (predicted == y_train).sum().item()    
    total = y_train.size(0)                           
    accuracy = correct / total * 100           


    if i % 10 == 0:
        print(f"Epoch: {i} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")


# ======= Avaliação no teste =======
with torch.no_grad():
    outputs_test = model(X_test)
    _, predicted_test = torch.max(outputs_test, 1)
    correct_test = (predicted_test == y_test).sum().item()
    accuracy_test = correct_test / y_test.size(0) * 100

print(f"\nTest Accuracy: {accuracy_test:.2f}%")

#======== Salvando o modelo se a acurácia for maior que 85% =======
if accuracy_test > 85:
    print("Model is good")
    print("Saving the model...")
    torch.save(model.state_dict(), "model_weights.pth")