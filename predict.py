import pickle
import pandas as pd

# load everything
model = pickle.load(open("model.pkl", "rb"))
ss = pickle.load(open("standard_scaler.pkl", "rb"))
mms = pickle.load(open("minmax_scaler.pkl", "rb"))

# column groups (based on your pipeline)
skewed_cols = ['person_income', 'loan_percent_income']
norm_cols = ['loan_int_rate']

def predict_loan(data_dict):
    df = pd.DataFrame([data_dict])
    
    # apply scaling
    df[skewed_cols] = ss.transform(df[skewed_cols])
    df[norm_cols] = mms.transform(df[norm_cols])
    
    prediction = model.predict(df)
    
    return int(prediction[0])