from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io, base64
from sklearn.datasets import load_iris

# ===== Modelo =====
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

in_feature = 4
hidden_layer = 10
out_feature = 3

model = PML(in_feature, hidden_layer, out_feature)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# ===== Carregando scaler do arquivo =====
scaler = joblib.load("scaler.save")

# ===== Carregando dataset original =====
iris = load_iris()
X_dataset = iris['data'][:, :2]  # vamos usar as duas primeiras features para plot
y_dataset = iris['target']

class_names = ["setosa", "versicolor", "virginica"]

# ===== FastAPI =====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # ou "*" para todas origens
    allow_methods=["*"],
    allow_headers=["*"],
)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float  

@app.post("/predict")
def predict(data: IrisData):
    try:
        # Transformando input com scaler salvo
        input_raw = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        input_scaled = scaler.transform(input_raw)  # <-- usa o scaler.save
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # Inferência
        with torch.no_grad():   
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_name = class_names[predicted_class]
        
        # ===== Gerando gráfico =====
        plt.figure(figsize=(6,5))
        for cls in np.unique(y_dataset):
            plt.scatter(X_dataset[y_dataset==cls,0], X_dataset[y_dataset==cls,1], label=class_names[cls])
        plt.scatter(input_raw[0,0], input_raw[0,1], color='red', s=100, marker='*', label='New Sample')
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "predicted_class": predicted_class,
            "predicted_name": predicted_name,
            "plot_base64": img_base64
        }
    except Exception as e:
        return {"error": str(e)}
    
