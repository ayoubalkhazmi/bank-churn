from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd, joblib

app = FastAPI()

# Load the Brain
data = joblib.load("churn_model.pkl")
model = data['model']
scaler = data['scaler']
encoders = data['encoders']
feature_names = data['feature_names']

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Churn Predictor</title>
            <style>
                body{font-family:sans-serif;max-width:500px;margin:auto;padding:20px;}
                label{display:block;margin-top:10px;font-weight:bold;}
                input,select{width:100%;padding:8px;margin-top:5px;}
                button{margin-top:20px;width:100%;padding:10px;background:#d9534f;color:white;border:none;cursor:pointer;}
            </style>
        </head>
        <body>
            <h2>🏦 Bank Churn Risk (DT Depth 3)</h2>
            <form action="/predict" method="post">
                <label>Credit Score:</label><input type="number" name="CreditScore" value="600" required>
                <label>Geography:</label>
                <select name="Geography">
                    <option value="France">France</option>
                    <option value="Spain">Spain</option>
                    <option value="Germany">Germany</option>
                </select>
                <label>Gender:</label>
                <select name="Gender"><option value="Male">Male</option><option value="Female">Female</option></select>
                <label>Age:</label><input type="number" name="Age" value="40" required>
                <label>Tenure (Years):</label><input type="number" name="Tenure" value="3" required>
                <label>Balance:</label><input type="number" step="0.01" name="Balance" value="60000" required>
                <label>Number of Products:</label><input type="number" name="NumOfProducts" value="1" required>
                <label>Has CrCard (1=Yes, 0=No):</label><input type="number" name="HasCrCard" value="1" required>
                <label>Is Active Member (1=Yes, 0=No):</label><input type="number" name="IsActiveMember" value="1" required>
                <label>Estimated Salary:</label><input type="number" step="0.01" name="EstimatedSalary" value="50000" required>
                <button type="submit">Predict Risk</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(
    CreditScore: int = Form(...), Geography: str = Form(...), Gender: str = Form(...),
    Age: int = Form(...), Tenure: int = Form(...), Balance: float = Form(...),
    NumOfProducts: int = Form(...), HasCrCard: int = Form(...),
    IsActiveMember: int = Form(...), EstimatedSalary: float = Form(...)
):
    try:
        # 1. Create a "Clean" dictionary of your inputs (No spaces, lowercase)
        # This acts as a lookup table.
        form_inputs = {
            'creditscore': CreditScore,
            'geography': Geography,
            'gender': Gender,
            'age': Age,
            'tenure': Tenure,
            'balance': Balance,
            'numofproducts': NumOfProducts,
            'pointsearned': NumOfProducts, # Handle alias
            'hascrcard': HasCrCard,
            'creditcard': HasCrCard,       # Handle alias
            'isactivemember': IsA
