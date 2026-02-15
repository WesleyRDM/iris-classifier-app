Iris Predictor Web App
Description
This project is an Iris flower classifier using PyTorch for the machine learning model, FastAPI for the backend, and HTML/JavaScript for the front-end.
The application allows users to enter the measurements of a flower (Sepal and Petal) and receive:
The predicted class (setosa, versicolor, or virginica)
A plot showing the position of the new sample relative to the original dataset
The project also uses a saved StandardScaler to ensure that user input is correctly normalized.

Technologies
Python 3.10+
PyTorch
Scikit-learn (Iris dataset, StandardScaler)

How to run locally
Clone the repository
git clone https://github.com/your-username/iris-predictor.git
cd iris-predictor
Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
Run the FastAPI API
uvicorn app:app --reload

The API will be available at: http://127.0.0.1:8000
