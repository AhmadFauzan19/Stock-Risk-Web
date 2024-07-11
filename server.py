import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
import plotly.graph_objs as go
matplotlib.use('Agg')  # Atur Matplotlib ke mode non-interaktif
import numpy as np   #Linear algera Library
import pandas as pd
import matplotlib.pyplot as plt  #to plot graphs
import seaborn as sns  #to plot graphs
from sklearn.linear_model import LinearRegression   #for linear regression model
sns.set()  #setting seaborn as default 
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# App title
st.markdown('''
# Stock Price App
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
df = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(df)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(df,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

print(df)
print(df.columns)

####
#st.write('---')
#st.write(tickerData.info)

stock_data = df

# Streamlit app
# Title
st.title('Prediksi Harga Saham Berikutnya dengan Regresi Linear')

# Show stock data table
st.subheader('Data Historis Saham')
st.write(stock_data)

# Select columns for regression
column_options = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
selected_column = st.selectbox('Pilih kolom untuk melakukan regresi linear', column_options)

# Perform linear regression
X = stock_data['Open'].values.reshape(-1, 1)
Y = stock_data[selected_column].values
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
r2 = model.score(X, Y)  # Using model.score to get R-squared

# Plot regression line and data points
fig, ax = plt.subplots()
ax.scatter(stock_data['Open'], stock_data[selected_column], label='Data Asli')
ax.plot(stock_data['Open'], Y_pred, color='red', label='Regresi Linear')
ax.set_xlabel('Open Price')
ax.set_ylabel(selected_column)
ax.legend()

# Predict next day's stock price
next_day_open = stock_data.iloc[-1]['Open']
next_day_predicted_close = model.predict([[next_day_open]])

# Show predictions
st.subheader('Prediksi Harga Saham Besok')
st.write(f'Open Price: {next_day_open}')
st.write(f'Predicted {selected_column}: {next_day_predicted_close[0]:.2f}')

# Plot predicted next day's stock price
fig_next_day, ax_next_day = plt.subplots()
ax_next_day.plot(stock_data.index, stock_data[selected_column], label='Data Asli')
ax_next_day.plot(stock_data.index[-1] + pd.DateOffset(days=1), next_day_predicted_close[0], 'ro', label='Prediksi Besok')
ax_next_day.set_xlabel('Date')
ax_next_day.set_ylabel(selected_column)
ax_next_day.legend()

# Show plot in Streamlit
st.subheader(f'Prediksi Harga Saham {selected_column} Besok')
st.pyplot(fig_next_day)