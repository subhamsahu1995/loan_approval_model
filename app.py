from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 USER INPUTS
        income = float(request.form['income'])
        loan = float(request.form['loan'])
        rate = float(request.form['rate'])
        credit = float(request.form['credit'])
        default = int(request.form['default'])
        emp_exp = float(request.form['emp_exp'])
        cred_hist = float(request.form['cred_hist'])

        # 🔥 AUTO FEATURE (USER DOES NOT ENTER THIS)
        loan_percent = loan / income

        # 🔹 FINAL INPUT ORDER (MUST MATCH TRAINING)
        data = np.array([[
            income,
            loan,
            rate,
            loan_percent,
            credit,
            default,
            emp_exp,
            cred_hist
        ]])

        # 🔹 SCALE
        data_scaled = scaler.transform(data)

        # 🔹 PROBABILITY
        prob = model.predict_proba(data_scaled)[0][1]

        # 🔥 DECISION LOGIC (FIXED)
        if prob >= 0.7:
            result = "Approved ✅"
            risk = "Low Risk"
        elif prob >= 0.4:
            result = "Borderline ⚠️"
            risk = "Moderate Risk"
        else:
            result = "Rejected ❌"
            risk = "High Risk"

        return render_template(
            'index.html',
            prediction=result,
            probability=round(prob * 100, 2),
            risk=risk
        )

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)