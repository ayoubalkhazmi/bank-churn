from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd, joblib

app = FastAPI()

# --- LOAD THE BRAIN ---
data = joblib.load("churn_model.pkl")
model = data['model']
scaler = data['scaler']
encoders = data['encoders']
feature_names = data['feature_names']

# --- HOME PAGE ---
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
                button{margin-top:20px;width:100%;padding:10px;background:#d9534f;color:white;border:none;}
            </style>
        </head>
        <body>
            <h2>🏦 Bank Churn Risk (DT Depth 3)</h2>
            <form action="/predict" method="post">
                <label>Credit Score:</label>
                <input type="number" name="CreditScore" value="600" required>
                
                <label>Geography:</label>
                <select name="Geography">
                    <option value="France">France</option>
                    <option value="Spain">Spain</option>
                    <option value="Germany">Germany</option>
                </select>
                
                <label>Gender:</label>
                <select name="Gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
                
                <label>Age:</label>
                <input type="number" name="Age" value="40" required>
                
                <label>Tenure (Years):</label>
                <input type="number" name="Tenure" value="3" required>
                
                <label>Balance:</label>
                <input type="number" step="0.01" name="Balance" value="60000" required>
                
                <label>Number of Products:</label>
                <input type="number" name="NumOfProducts" value="1" required>
                
                <label>Has CrCard (1=Yes, 0=No):</label>
                <input type="number" name="HasCrCard" value="1" required>
                
                <label>Is Active Member (1=Yes, 0=No):</label>
                <input type="number" name="IsActiveMember" value="1" required>
                
                <label>Estimated Salary:</label>
                <input type="number" step="0.01" name="EstimatedSalary" value="50000" required>
                
                <button type="submit">Predict Risk</button>
            </form>
        </body>
    </html>
    """

# --- PREDICTION LOGIC ---
@app.post("/predict", response_class=HTMLResponse)
def predict(
    CreditScore: int = Form(...), Geography: str = Form(...), 
    Gender: str = Form(...), Age: int = Form(...), 
    Tenure: int = Form(...), Balance: float = Form(...),
    NumOfProducts: int = Form(...), HasCrCard: int = Form(...),
    IsActiveMember: int = Form(...), EstimatedSalary: float = Form(...)
):
    try:
        # 1. Clean Inputs Lookup Table
        # (Lowercase, no spaces)
        form_inputs = {
            'creditscore': CreditScore,
            'geography': Geography,
            'gender': Gender,
            'age': Age,
            'tenure': Tenure,
            'balance': Balance,
            'numofproducts': NumOfProducts,
            'pointsearned': NumOfProducts, 
            'hascrcard': HasCrCard,
            'creditcard': HasCrCard,       
            'isactivemember': IsActiveMember,
            'active': IsActiveMember,      
            'estimatedsalary': EstimatedSalary,
            'salary': EstimatedSalary      
        }

        # 2. Match Model Columns
        final_input_dict = {}
        
        for model_col in feature_names:
            # Clean model col name: "Credit Score" -> "creditscore"
            clean_col = model_col.replace(" ", "").replace("_", "").lower()
            
            if clean_col in form_inputs:
                final_input_dict[model_col] = [form_inputs[clean_col]]
            else:
                return f"Error: Model expects '{model_col}' but form doesn't have it."

        df = pd.DataFrame(final_input_dict)
        
        # 3. Apply Encoders
        for col, le in encoders.items():
            if col in df.columns: 
                df[col] = le.transform(df[col])
            
        # 4. Scale and Predict
        X_final = scaler.transform(df)
        pred = model.predict(X_final)[0]
        
        # Result Formatting
        msg = "⚠️ CUSTOMER WILL LEAVE" if pred == 1 else "✅ CUSTOMER WILL STAY"
        color = "red" if pred == 1 else "green"
        
        # We use triple quotes here to prevent "unterminated string" errors
        return f"""
        <html>
            <body style='text-align:center;font-family:sans-serif;'>
                <h1 style='color:{color}'>{msg}</h1>
                <a href='/'>Back</a>
            </body>
        </html>
        """
        
    except Exception as e: 
        return f"Error: {e}"
