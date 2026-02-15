Iris Predictor Web App
Descrição
Este projeto é um classificador de flores Iris utilizando PyTorch para o modelo de machine learning, FastAPI para o backend e HTML/JavaScript para o front-end.
O aplicativo permite que o usuário insira as medidas de uma flor (Sepal e Petal) e receba:
A classe prevista (setosa, versicolor ou virginica)
Um gráfico mostrando a posição da nova amostra em relação ao dataset original
O projeto também utiliza um scaler salvo (StandardScaler) para garantir que os dados do usuário sejam normalizados corretamente.


Tecnologias
Python 3.10+
PyTorch
Scikit-learn (dataset Iris, StandardScaler)
FastAPI
Matplotlib (para gerar gráficos)
HTML + JavaScript para front-end


Como rodar localmente
Clonar o repositório
git clone https://github.com/seu-usuario/iris-predictor.git
cd iris-predictor
Criar um ambiente virtual e instalar dependências
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
Rodar a API FastAPI
uvicorn app:app --reload
