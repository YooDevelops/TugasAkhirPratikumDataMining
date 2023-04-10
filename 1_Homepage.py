import streamlit as st
from pathlib import Path
#from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Path setting
#current_dir = Path(__file__).parent if "_file_" in locals() else Path.cwd()
#profile_pict1 = current_dir / "Prasetyo.png"
#profile_pict2 = current_dir / "Pando.png"
#profile_pict3 = current_dir / "Firdos.png"


st.set_page_config(
    page_title="Multiple App"
)
st.markdown(
    "<div style='text-align:center;position:relative;top:-5px;font-size:3rem'>Kelompok Tugas Akhir Pratikum Data Mining</div>",
    unsafe_allow_html=True)
st.sidebar.success("Select page above.")


st.write('')
st.write('')
st.write('')
st.write('')

profile_pict1 = Image.open(profile_pict1)
profile_pict2 = Image.open(profile_pict2)
profile_pict3 = Image.open(profile_pict3)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(profile_pict1, width=200)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>Prasetyo</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>312010126</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>TI.20.C1</div>",
        unsafe_allow_html=True)

with col2:
    st.image(profile_pict2, width=200)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>Fani Trifando</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>312010445</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>TI.20.C1</div>",
        unsafe_allow_html=True)

with col3:
    st.image(profile_pict3, width=200)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>M. Ilham Firdaus</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>312010313</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;position:relative;top:-5px;font-size:2rem'>TI.20.C1</div>",
        unsafe_allow_html=True)
st.write("\n")  # memberikan jarak antara row

header = st.container()

with header:
    st.header('Latar Belakang')
    st.markdown(
        "<div style='text-align:justify;position:relative;top:-5px;font-size:1rem'>Dalam projek ini kami memilih judul \"Prediksi Harga Mobil Bekas Audi di United Kingdom. \"</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:justify;position:relative;top:-5px;font-size:1x`rem'>Kami Memilih judul tersebut untuk menganalisa dataset tersebut dan adapun Pemilihan mobil bekas Audi sebagai fokus dalam projek ini dapat disebabkan oleh beberapa alasan, antara lain: <div>",
        unsafe_allow_html=True)
    st.markdown("""
1. Audi merupakan salah satu merek mobil premium yang cukup populer di pasaran. Mobil Audi memiliki fitur dan teknologi canggih yang memungkinkan pengendara untuk merasakan kenyamanan, keamanan, dan performa mesin yang handal.
2. Meskipun harga mobil Audi tergolong mahal, namun mobil ini memiliki nilai jual yang cukup tinggi pada saat dijual kembali. Hal ini disebabkan karena mobil Audi memiliki daya tahan dan kualitas yang baik serta terkenal dengan performa mesin yang handal.
3. Pasar mobil bekas Audi cukup menjanjikan karena mobil Audi dikenal memiliki nilai jual yang stabil. Selain itu, permintaan mobil bekas Audi juga cukup tinggi karena faktor-faktor seperti performa mesin yang handal, fitur dan teknologi canggih, serta model mobil yang bervariasi.
""")

    st.text('Dan kami mengambil dataset dari Kaggle : ')
    st.write("[https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)")

    st.markdown(
        "<hr>", unsafe_allow_html=True)

    # Menambahkan teks setelah garis bawah
    st.header("Menggunakan Metode Linear Regression")
    st.markdown(
        "<div style='text-align:justify;position:relative;top:-5px;font-size:1rem'> Linear regression (regresi linear) adalah salah satu metode dalam statistika yang digunakan untuk memodelkan hubungan antara sebuah variabel independen (biasanya disebut x) dengan sebuah variabel dependen (biasanya disebut y) secara linier. Tujuan dari linear regression adalah untuk menemukan persamaan garis lurus (linear) yang paling sesuai dengan data yang ada.</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)


@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)
    return df


df = get_data('audi.csv')

# Define the features and target variable
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
X = df[features]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=70)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Calculate the accuracy score of the model
score = lr.score(X_test, y_test)
score_percent = round(score * 100, 2)

# Create a pie chart with the score percentage
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([score_percent, 100-score_percent],
       labels=[f"Score: {score_percent}%", " "])
ax.set_title("Persentase Skor Akurasi Model Regresi Linear Pada Datasheet")
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)

st.header('Informasi Dataset')
# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# menghitung jumlah baris dan kolom
jumlah_data = df.shape

# mencetak jumlah data
st.write('Jumlah data:', jumlah_data)
st.write('Jumlah data dalam X_train :', X_train.shape[0])
st.write('Jumlah data dalam X_test  :', X_test.shape[0])
st.write('Jumlah data dalam y_train :', y_train.shape[0])
st.write('Jumlah data dalam y_test  :', y_test.shape[0])
