import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the model
with open(r'C:\Users\owner\Desktop\Prediction_Model\Price_PredictorML\price_prediction_model (2).pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
train = pd.read_csv(r'C:\Users\owner\Desktop\Prediction_Model\Price_PredictorML\test_data - Flipkart_Mobiles.csv')
train['Date'] = pd.to_datetime(train['Date'], format="%d %b %Y")
train = train.sort_values(by='Date').reset_index(drop=True)
train = train.set_index("Date")
train['Product'] = train['Brand'] + ' ' + train['Model']

def train_model(product):
    product_data = train[train['Product'] == product].sort_values('Date')

    if product_data.empty:
        return None, None

    # Create lag features
    product_data['Selling_Price_Lag1'] = product_data['Selling Price'].shift(1)
    product_data['Selling_Price_Lag2'] = product_data['Selling Price'].shift(2)
    product_data = product_data.dropna()

    if product_data.shape[0] < 3:
        return None, None

    X = product_data[['Selling_Price_Lag1', 'Selling_Price_Lag2']]
    y = product_data['Selling Price']

    model = LinearRegression()
    model.fit(X, y)

    return model, product_data

def forecast_next_days(model, product_data, days=2):
    last_data = product_data[['Selling_Price_Lag1', 'Selling_Price_Lag2']].iloc[-1].values
    forecasts = []

    for _ in range(days):
        next_price = model.predict([last_data])[0]
        forecasts.append(next_price)
        last_data = np.array([next_price, last_data[0]])

    return forecasts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    product = request.form['product']
    model, product_data = train_model(product)
    
    if model and product_data is not None:
        future_prices = forecast_next_days(model, product_data, days=2)
        forecasts = [(i + 1, price) for i, price in enumerate(future_prices)]
        return render_template('index.html', forecasts=forecasts)
    else:
        return render_template('index.html', forecasts=None, error="Product data not found or insufficient data to make predictions.")

if __name__ == '__main__':
    app.run(debug=True)
