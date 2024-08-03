# Price Prediction Model

This project is a simple web application that predicts future prices of mobile products based on historical data. The app is built using Flask, with a linear regression model for prediction. The user can input a product name, and the app will display forecasted prices for the next two days.

## Description

The Price Prediction Model uses historical selling prices of mobile products to predict future prices. The application is built with Flask for the web interface and scikit-learn for the machine learning model. It leverages pandas for data manipulation and numpy for numerical operations.

## Steps to Run the Project

1. **Create a virtual environment:**

    ```bash
    python -m venv myenv
    ```

2. **Activate the virtual environment:**

    On Windows:
    ```bash
    myenv\Scripts\activate
    ```

    On macOS/Linux:
    ```bash
    source myenv/bin/activate
    ```

3. **Install required packages:**

    ```bash
    pip install flask pandas==2.1.4 numpy==1.26.4 scikit-learn==1.2 gunicorn
    ```

4. **Upgrade pip (optional but recommended):**

    ```bash
    python.exe -m pip install --upgrade pip
    ```

5. **Run the Flask application:**

    ```bash
    python app.py
    ```

## Directory Structure
flask_app/
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── price_prediction_model.pkl
├── app.py
└── test_data - Flipkart_Mobiles.csv


## Example Image


![WhatsApp Image 2024-08-03 at 19 24 28_97de91fa](https://github.com/user-attachments/assets/2936d07e-6cb4-4b7d-95c5-ec07f6eb273d)

## Dataset Link

**Access Link:** https://drive.google.com/file/d/15Knb-EZTVrsEtB9S10Wy5lT4VzgNSe2u/view?usp=sharing





