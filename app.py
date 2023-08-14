from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Define the columns used in the training data
# Make sure the column order matches the one used during training
columns = ['Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

# Load the trained model from the pkl file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize label encoders for categorical columns
categorical_columns = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Load training data to fit label encoders
# Replace this with actual training data
training_data = pd.read_csv('E:\PROJECTS\LOAN APPROVAL PREDICTION\Training Data.csv')  # Provide the path to your training data CSV file
for col in categorical_columns:
    label_encoders[col].fit(training_data[col])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'Income': int(request.form['Income']),
            'Age': int(request.form['Age']),
            'Experience': int(request.form['Experience']),
            'Married/Single': request.form['Married/Single'],
            'House_Ownership': request.form['House_Ownership'],
            'Car_Ownership': request.form['Car_Ownership'],
            'Profession': request.form['Profession'],
            'CITY': request.form['CITY'],
            'STATE': request.form['STATE'],
            'CURRENT_JOB_YRS': int(request.form['CURRENT_JOB_YRS']),
            'CURRENT_HOUSE_YRS': int(request.form['CURRENT_HOUSE_YRS'])
        }

        # Preprocess user input using label encoders
        for col in categorical_columns:
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Convert the preprocessed user input to DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Ensure columns match the training data columns
        user_input_df = user_input_df[columns]

        # Predict using the loaded model
        predicted_risk_flag = model.predict(user_input_df)
        predicted_result = 'Yes' if predicted_risk_flag[0] == 1 else 'No'

        return render_template('result.html', result=predicted_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
