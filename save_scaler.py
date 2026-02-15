from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import joblib


X, y = load_iris(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X) 

joblib.dump(scaler, "scaler.save")

print("Scaler salvo com sucesso!")