# Loan Approval Prediction App

This repository contains a simple web application that predicts loan approval using a machine learning model built with the Decision Tree Classifier algorithm. The application takes input details from users through a web interface and uses the trained model to predict whether a loan application is likely to be approved or rejected.

## Table of Contents
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data](#data)
- [Summary](#summary)

## Project Structure

- `static/`: Contains the necessary styles for the HTML templates.
- `template/`: Contains the HTML templates for the application.
  - `index.html`: Main template for user input.
  - `result.html`: Displayed when the loan approval has a high chance of approval.
  - `results.html`: Displayed when the loan rejection chance is higher.
- `model.py`: Trains the machine learning model and saves it as a `.pkl` file.
- `app.py`: Implements Flask connectivity to serve the web application.
- `data/`: Directory for storing CSV files used for model training and testing.

## How It Works

1. Users input loan application details through the `index.html` page.
2. The `app.py` file processes the user input using Flask.
3. The input data is converted into a DataFrame and passed to the trained machine learning model.
4. The model predicts whether the loan will be approved or rejected.
5. Depending on the prediction, either the `result.html` or `results.html` page is displayed.

## Getting Started

To run the application locally:

1. Clone this repository to your machine.
2. Make sure you have the necessary dependencies (see [Dependencies](#dependencies)).
3. Run the `app.py` file using your preferred Python environment.
4. Access the application through your web browser at the provided local server link.

## Dependencies

- Python (>= 3.6)
- Flask
- pandas
- scikit-learn

Install the required dependencies using the following command:

'''bash
pip install flask pandas scikit-learn
''''


# Usage

Run the app.py file.
Open your web browser and navigate to the local server link provided in the terminal.
Fill in the loan application details in the provided form.
Submit the form to see the prediction result.
Data
The machine learning model is trained using data from Kaggle. The CSV files used for training and testing are stored in the data/ directory.

# Summary

This project demonstrates a loan approval prediction application that utilizes a Decision Tree Classifier machine learning model. The application takes user input through a web interface, processes it using Flask, and predicts loan approval status. The model is trained using data from Kaggle, and the trained model is saved as a .pkl file. The application showcases the interaction between HTML templates and Python using Flask.
