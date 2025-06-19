import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

app = Flask(__name__)

# Set up a folder to save uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return "No selected file or invalid file type. Please upload a CSV or Excel file."

    # Read the dataset
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        if filename.endswith('.csv'):
            data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        elif filename.endswith(('.xls', '.xlsx')): 
            data = pd.read_excel(filename, parse_dates=['Date'], index_col='Date')

        # Ensure daily frequency and forward fill missing values
        data = data.asfreq('D')
        data.fillna(method='ffill', inplace=True)

        # Stationarity test
        result = adfuller(data['Close'])
        adf_stat = result[0]
        p_value = result[1]

        # Fit SARIMA model
        model = auto_arima(data['Close'], seasonal=True, m=12)  
        order = (1, 1, 1) 
        seasonal_order = (1, 1, 1, 12)  
        
        sarima_model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
        sarima_results = sarima_model.fit()
        
        # Forecasting future values with SARIMA
        n_periods = 30
        forecast_sarima = sarima_results.get_forecast(steps=n_periods)
        forecast_index_sarima = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_periods)
        forecast_values_sarima = forecast_sarima.predicted_mean

        # Fit ARIMA model using Close price
        arima_order = (1, 1, 1)  # (p, d, q)
        arima_model_close = ARIMA(data['Close'], order=arima_order)
        arima_results_close = arima_model_close.fit()

        # Forecasting future values with ARIMA
        forecast_arima = arima_results_close.get_forecast(steps=n_periods)
        forecast_index_arima = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_periods)
        forecast_values_arima = forecast_arima.predicted_mean

        # Linear Regression Model
        data['Days'] = (data.index - data.index[0]).days 
        X = data[['Days']] 
        y = data['Close']   

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a linear regression model
        linreg_model = LinearRegression()

        # Fit the model on training data
        linreg_model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = linreg_model.predict(X_test)

        # Calculate accuracy (MAPE)
        accuracy = calculate_accuracy(y_test, y_pred)

        # Make predictions on the entire dataset for plotting purposes
        y_full_pred = linreg_model.predict(X)

        # Save the trained model and dataset using joblib
        joblib.dump(linreg_model, 'linearregression.joblib')
        joblib.dump(data, 'stock_data.joblib')

        # Plotting the SARIMA forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Historical Prices (SARIMA)', color='blue')
        plt.plot(forecast_index_sarima, forecast_values_sarima, label='Forecasted Prices (SARIMA)', color='red')
        plt.fill_between(forecast_index_sarima,
                        forecast_sarima.conf_int()['lower Close'],
                        forecast_sarima.conf_int()['upper Close'], color='pink')
        plt.title('SARIMA Stock Price Forecast')
        plt.legend(loc="best")

        # show date in Month-Year format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)

        # simpan jd gambar
        img_sarima = io.BytesIO()
        plt.savefig(img_sarima, format='png')
        img_sarima.seek(0)

        plot_url_sarima = base64.b64encode(img_sarima.getvalue()).decode()
        plt.close()

        # Plotting the results for ARIMA
        plt.figure(figsize=(12, 6))
        plt.plot(data['Adj Close'], label='Historical Prices (ARIMA)', color='blue')
        plt.plot(forecast_index_arima, forecast_values_arima, label='Forecasted Prices (ARIMA)', color='green')
        plt.fill_between(forecast_index_arima,
                         forecast_arima.conf_int()['lower Close'],
                         forecast_arima.conf_int()['upper Close'], color='lightgreen')
        plt.title('ARIMA Stock Price Forecast')
        plt.legend(loc="best")

        # show date in Month-Year format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        img_arima = io.BytesIO()
        plt.savefig(img_arima, format='png')
        img_arima.seek(0)
        plot_url_arima = base64.b64encode(img_arima.getvalue()).decode()
        plt.close()

        # Plotting the results for Linear Regression
        plt.figure(figsize=(12, 6)) 
        plt.plot(data.index, data['Close'], label='Actual Prices', color='blue', linewidth=2)
        plt.scatter(X_test.index, y_test, color='orange', label='Test Data', alpha=0.6, edgecolor='k')
        plt.plot(data.index, y_full_pred, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.title('Price Prediction Using Linear Regression', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price (Close)', fontsize=12)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        
        img_linreg = io.BytesIO()
        plt.savefig(img_linreg, format='png')
        img_linreg.seek(0)
        plot_url_linreg = base64.b64encode(img_linreg.getvalue()).decode()

        return render_template('forecast.html', adf_stat=adf_stat, p_value=p_value,
                               plot_url_sarima=plot_url_sarima, plot_url_arima=plot_url_arima,
                               plot_url_linreg=plot_url_linreg, accuracy=accuracy)

    return "Something went wrong. Please try again."

def calculate_accuracy(y_actual, y_predicted):
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    accuracy = 100 - mape
    return accuracy

if __name__ == '__main__':
    app.run(debug=True)
