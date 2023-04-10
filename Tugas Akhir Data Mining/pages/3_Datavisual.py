import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
interactive = st.container()


@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)

    return df


with header:
    st.header("Visualisasi Data Harga Mobil Bekas Audi di United Kingdom")
    st.markdown("<hr>", unsafe_allow_html=True)

with dataset:
    df = get_data('audi.csv')

    st.subheader('Model Mobil yang tersedia')
    model = pd.DataFrame(df['model'].value_counts().head(50))
    st.bar_chart(model)

    st.subheader('Harga Mobil Dengan BBM Tertentu')
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='price', hue='fuelType',
                 fill=False, element='step', ax=ax)
    ax.set_title("Harga mobil dengan jenis bahan bakar tertentu")
    ax.set_xlabel("Harga (Pounds)")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    st.subheader('Model Mobil yang tersedia')
    model = pd.DataFrame(df['model'].value_counts().head(50))
    st.bar_chart(model)

    st.subheader('Harga Mobil Berdasarkan Umur')
    fig, ax = plt.subplots()
    sns.lineplot(x="year", y="price", data=df)
    plt.title("Harga mobil berdasarkan jumlah tahun")
    plt.xlabel("Tahun")
    plt.ylabel("Harga (Pounds)")
    st.pyplot(fig)

    st.subheader('Banyak Jenis Bahan Bakar')

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the countplot
    sns.countplot(x="fuelType", data=df, ax=ax)

    # Set the plot title
    ax.set_title("Jenis Bahan Bakar dari Mobil yang terdaftar")

    # Show the plot
    st.pyplot(fig)


with features:
    st.subheader("Rata-rata Harga Mobil Berdasarkan Model")
    fig, ax = plt.subplots(figsize=(18, 18))
    priceByModel = df.groupby("model")['price'].mean().reset_index()
    ax.set_title("Average Price of Vehicle")
    sns.set()
    sns.barplot(x='model', y='price', data=priceByModel, ax=ax)
    plt.xticks(rotation=60)
    st.pyplot(fig)


st.subheader('Plot Histogram Attribut')
fig, axes = plt.subplots(figsize=(12, 10), nrows=2, ncols=3)

sns.histplot(df["year"], ax=axes[0, 0])
sns.histplot(df["mileage"], ax=axes[0, 1])
sns.histplot(df["tax"], ax=axes[0, 2])
sns.histplot(df["mpg"], ax=axes[1, 0])
sns.histplot(df["engineSize"], ax=axes[1, 1])
sns.histplot(df["price"], ax=axes[1, 2])

st.pyplot(fig)


st.subheader('Korelasi antar attribute dalam Matrix / Heatmap')
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(df.corr(), cmap="Reds", annot=True, ax=ax)
plt.title("Correlation HeatMap/ Matrix")

st.pyplot(fig)
