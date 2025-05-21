from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the calibrated model, threshold, and other necessary files
calibrated_model = joblib.load('calibrated_model.pkl')  # Load your calibrated model
with open('selected_threshold.pkl', 'rb') as f:
    selected_threshold = pickle.load(f)  # Load the selected threshold value
with open('training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)  # Load training columns

# Function to preprocess input data to match training format
def preprocess_input(df):
    """Preprocess the input exactly as during training."""
    # Preprocessing steps similar to those used during training
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 
                                                    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Convert numeric columns
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Reindex to match training columns
    df = df.reindex(columns=training_columns, fill_value=0)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form input
    input_data = request.form.to_dict()
    customer_id = input_data.pop("customerID")  # Get Customer ID from form
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data to match the training format
    input_df_processed = preprocess_input(input_df)
    
    # Predict probabilities using the calibrated model
    pred_proba = calibrated_model.predict_proba(input_df_processed)[0][1]
    
    # Apply threshold to classify as Churn or Not Churn
    prediction = "Churn" if pred_proba >= selected_threshold else "Not Churn"
    
    # Return the result to the result page
    return render_template('result.html', customer_id=customer_id, prediction=prediction, probability=round(pred_proba * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
