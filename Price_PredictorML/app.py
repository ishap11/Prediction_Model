from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from the new pickle file
with open(r'C:\Users\owner\Desktop\render\price_prediction_model .pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data (assuming the data is available in the same directory)
train = pd.read_csv(r'C:\Users\owner\Desktop\render\train_dataset.csv')
train['Date'] = pd.to_datetime(train['Date'], format="%d/%m/%Y")
train = train.sort_values(by='Date').reset_index(drop=True)
train = train.set_index("Date")

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        product_id = request.form['product_id']
        prediction = predict_product_price(product_id)
    return render_template('index.html', prediction=prediction)

def predict_product_price(product_id, alpha=0.9, future_days=2):
    # Filter data for the specified product
    product_data = train[train['Product'] == product_id].sort_values('Date')

    if product_data.empty:
        return "No data found for this product."

    # Create lag features for the product data
    product_data['Selling_Price_Lag1'] = product_data['Selling_Price'].shift(1)
    product_data['Selling_Price_Lag2'] = product_data['Selling_Price'].shift(2)
    product_data = product_data.dropna()

    X = product_data[['Selling_Price_Lag1', 'Selling_Price_Lag2']]
    predictions = model.predict(X)
    return predictions[-1]

if __name__ == '__main__':
    app.run(debug=True)
