import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)

    return df


df = get_data('audi.csv')
# Split data into features and labels
X = df.drop(columns=['year', 'mileage', 'tax', 'mpg', 'engineSize'])
y = df['price']

# Define the features and target variable
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
X = df[features]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=70)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)


def predict_price(year, mileage, tax, mpg, engineSize):
    price = lr.predict([[year, mileage, tax, mpg, engineSize]])
    return price[0]


def gbp_to_idr(price):
    rate = 19500.00   # asumsi kurs 1 GBP = 19500 IDR
    return price * rate


# Create the Streamlit app
st.title('Prediksi Harga Mobil Bekas Audi di United Kingdom')
st.markdown('Aplikasi ini memprediksi harga mobil berdasarkan fitur-fiturnya menggunakan Linear Regresi.')

# Create sliders for user input
year = st.slider('Year', 1997, 2010, 2023)
mileage = st.slider('Mileage', 0, 2000, 50000)
tax = st.slider('Tax', 0, 2000, 1000)
mpg = st.slider('MPG', 10, 300, 30)
engineSize = st.slider('Engine Size', 0, 10, 2)


# Make predictions and display the result
if st.button('Predict'):
    price = predict_price(year, mileage, tax, mpg, engineSize)
    idr_price = gbp_to_idr(price)
    idr_price_formatted = 'Rp {:,.2f}'.format(idr_price).replace(
        ',', '|').replace('.', ',').replace('|', '.')
    st.success('Harga mobil yang diprediksi dalam Pounds : {:.2f} atau dalam IDR : {}'.format(
        price, idr_price_formatted))

