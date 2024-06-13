---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="W9IKLx9AwJzS" -->
# **SISTEM REKOMENDASI DESTINASI WISATA NoD (NEAR OF DESTINATION)**


---
Team ID : `C241-PS414`

<!-- #endregion -->

<!-- #region id="Ik87JblswY3p" -->
# Instalasi Libraries
**`Numpy` `Scikit-Learn` `Surprise` `Geodesic` `Pandas` `TensorFlow` `TensorFlowJs`**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Qxg6Lj_Zt-ov" outputId="3db048e5-55af-4d37-8163-b66f19945c2b"
!pip install surprise
!pip install tensorflow
!pip install tensorflowjs
!pip install pydantic-settings
!pip install pandas-profiling
```

```python colab={"base_uri": "https://localhost:8080/"} id="ahDsQ17MuF1o" outputId="bf7f99c9-ce00-40d2-f854-ae742abf5f59"
# Manipulasi dan Analisis Data
# pandas: Untuk manipulasi data dan analisis data.
# numpy: Untuk operasi numerik.
import pandas as pd
import numpy as np
import sklearn

# Visualisasi Data
# seaborn: Untuk visualisasi data statistik.
# matplotlib.pyplot: Untuk membuat plot dan grafik.
import seaborn as sns
import matplotlib.pyplot as plt

# Pembelajaran Mesin
# sklearn.preprocessing: Untuk encoding label dan standarisasi fitur.
# sklearn.model_selection: Untuk membagi dataset menjadi set pelatihan dan set pengujian.
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep Learning
# tensorflow.keras.models: Untuk membangun dan menyimpan model.
# tensorflow.keras.layers: Untuk membuat lapisan dalam model neural network.
# tensorflow.keras.callbacks: Untuk callback seperti early stopping.
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
# Geographical Calculations
# geopy.distance: Untuk menghitung jarak geografis antara koordinat.
from geopy.distance import geodesic

# Profiling dan Pelaporan Data
# ydata_profiling: Untuk membuat laporan profil data secara otomatis.
import ydata_profiling

# Interaksi Sistem dan File
# os: Untuk berinteraksi dengan sistem operasi, seperti membaca daftar file dalam direktori.
# shutil: Untuk mengoperasikan file dan koleksi file.
import os
import shutil

# Google Colab Integration
# google.colab.drive: Untuk mengakses Google Drive dari Colab.
# google.colab.files: Untuk mengunduh file dari Colab.
from google.colab import drive
from google.colab import files

# TensorFlow.js Conversion
# tensorflowjs: Untuk konversi model Keras ke format TensorFlow.js.
import tensorflowjs as tfjs

# Mengimpor warnings dan menonaktifkan pesan peringatan selama eksekusi kode
import warnings
warnings.filterwarnings("ignore")

import folium
from folium.plugins import MarkerCluster

# Menampilkan versi dari setiap library
print("Numpy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Pandas version:", pd.__version__)
print("TensorFlow version:", tf.__version__)
print("TensorFlow.js version:", tfjs.__version__)
```

<!-- #region id="XiuXg-u1xQuE" -->
# Data Wrangling
* **Gathering Data** = Melakukan pengumpulan data serta membaca dataset
* **Assessing Data** = Memeriksa dan memahami data
* **Cleaning Data** = Membersihkan data dari kesalahan/error
<!-- #endregion -->

<!-- #region id="nnBIfsMUxUPq" -->
## **Gathering Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FfzP4g8wxP0q" outputId="196e9549-9f42-4723-cc43-526fab60ce21"
# Mengakses dataset dari Google Drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/ML"
fnames = os.listdir(path)
print(fnames)
```

```python id="hVvW1XcmwE4Y"
# Load Dataset
NoD = pd.read_csv('/content/drive/MyDrive/ML/NoD.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="BQ0VB2AwyFGg" outputId="f280cebc-adda-4e66-d459-e54d1b5dff10"
# Menampilkan informasi ringkas tentang DataFrame
NoD.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ctjhKvP3ySpb" outputId="13f09367-ae54-495e-a1d1-39e596bec066"
# Menampilkan beberapa baris pertama dari DataFrame
NoD.head()
```

<!-- #region id="unjjEoNeyXHW" -->
## **Assessing Data**
<!-- #endregion -->

<!-- #region id="ymFMoWAL0uTw" -->
### Struktur Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oVaQTaKdyU6S" outputId="43c3bf65-10ef-418d-9061-9fa9a5313ddf"
print(NoD.shape)
```

<!-- #region id="EnpNv6X_0o8o" -->
### Kualitas Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DrFEgjOeyx_T" outputId="50b86002-618d-467f-b499-f579ba6c3f7e"
# Nilai yang hilang
print(NoD.isnull().sum())
```

```python colab={"base_uri": "https://localhost:8080/"} id="_qQyEa4Ry3Ms" outputId="d49c5adc-4577-4a0d-a5f8-799a8137292d"
# Duplikasi
print(NoD.duplicated().sum())
```

```python colab={"base_uri": "https://localhost:8080/"} id="VGuT2eyUy46p" outputId="ba3fd71b-28db-4c83-806b-86ba5cfe33a5"
# Ringkasan statistik untuk mendeteksi outlier
print(NoD.describe())
```

<!-- #region id="6sqD4bRg0hj2" -->
### Analisis Statistik Deskriptif
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="7IFtMrSIy98R" outputId="2eef1e05-415a-4acf-b231-3a4245f939b8"
# Histogram untuk kolom numerik
NoD.hist(bins=50, figsize=(20, 15))
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="FBm-lDC2zDGh" outputId="82d9bea6-6dfd-42d0-93a1-745b4540e9a6"
# Boxplot untuk mendeteksi outlier
sns.boxplot(data=NoD.select_dtypes(include=['float64', 'int64']))
plt.show()
```

<!-- #region id="1sctDHPhz0HO" -->
### Memahami Konten Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="K-cgIbZ5z5bk" outputId="e6e74800-9f8a-4224-c4c6-7142841ce45a"
print(NoD['Wilayah'].value_counts())
```

```python colab={"base_uri": "https://localhost:8080/"} id="tpy6i8otz-JN" outputId="f6a44f0f-992f-4067-fa29-85d367a2a068"
print(NoD['Provinsi'].value_counts())
```

```python colab={"base_uri": "https://localhost:8080/"} id="gBy8e6uM0DVS" outputId="5fee2450-a940-4d51-94e0-55e40fbaf111"
print(NoD['Kabupaten/Kota'].value_counts())
```

```python colab={"base_uri": "https://localhost:8080/"} id="uMar0yilzTFs" outputId="a12e5f6c-3037-4741-ee0f-479b7419b55f"
print(NoD['Jenis Wisata'].value_counts())
```

<!-- #region id="kYfgv1Bx0Tau" -->
### Distribusi Data untuk Setiap Fitur
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3Nc2WH3ZzfCn" outputId="fda51b85-ee1a-4130-8ec7-30d5e3ee686a"
for column in NoD.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.distplot(NoD[column].dropna(), kde=True)
    plt.title(f'Distribusi dari {column}')
    plt.show()
```

<!-- #region id="tidpwNw0098N" -->
### Analisis Kategorikal
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="YCF1EWf-zioO" outputId="353860df-a5b5-4f4c-b736-2b3eea22e542"
plt.figure(figsize=(10, 6))
sns.countplot(data=NoD, x='Jenis Wisata', order=NoD['Jenis Wisata'].value_counts().index)
plt.title('Distribusi Jenis Wisata')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="jCa5-p321B3_" outputId="7bc1f2e8-ee4b-46af-bce6-e1077edb8dc1"
plt.figure(figsize=(10, 6))
sns.barplot(data=NoD, x='Jenis Wisata', y='Rating')
plt.title('Rata-rata Rating per Jenis Wisata')
plt.show()
```

<!-- #region id="psaI8VgV1I0e" -->
### Analisis Geospasial
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 0} id="qDgRNDwj1LsM" outputId="462867d9-f945-402d-b989-169582c3a002"
# Membuat peta dasar
map_ = folium.Map(location=[-8.260630934632104, 115.39061833298781], zoom_start=10)

# Menambahkan titik-titik data ke peta
marker_cluster = MarkerCluster().add_to(map_)
for idx, row in NoD.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']],
                  popup=row['Nama Wisata']).add_to(marker_cluster)

map_
```

<!-- #region id="1ENXoOb2ZFXu" -->
### Mengecek Outlier
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="H2_6qs1OZMXv" outputId="5ad3a54f-9f56-42ec-c9ab-b34516c2c589"
# Menghitung Q1, Q3, dan IQR
Q1 = NoD['Reviews'].quantile(0.25)
Q3 = NoD['Reviews'].quantile(0.75)
IQR = Q3 - Q1

# Mengidentifikasi outlier
outliers_reviews = NoD[((NoD['Reviews'] < (Q1 - 1.5 * IQR)) | (NoD['Reviews'] > (Q3 + 1.5 * IQR)))]

# Mengetahui jumlah data yang outlier
reviews = len(outliers_reviews)
print(f"Jumlah data yang outlier reviews: {reviews}")
```

<!-- #region id="kx86uVhZ1ipu" -->
## Cleaning Data
<!-- #endregion -->

<!-- #region id="lDc1DXGH2Hhw" -->
### Menangani Nilai yang Hilang (Missing Values)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hvLBOfve1mo_" outputId="40b97bc0-8e62-4bed-c3f3-10963e453979"
# Menghapus baris yang memiliki nilai yang hilang
NoD.dropna(inplace=True)
print("Jumlah nilai hilang:", NoD.isnull().sum().sum())
```

<!-- #region id="L2NSoK_O2ekF" -->
### Menghapus Duplikasi Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mZ9KWHqH2iNU" outputId="100a3b47-3674-43a9-abbf-429f8157196d"
# Menghapus baris yang duplikat
NoD.drop_duplicates(inplace=True)
print("Jumlah nilai duplikat:", NoD.duplicated().sum())
```

<!-- #region id="scWu5X-g3Non" -->
### Memperbaiki Ketidakkonsistenan dalam Data
<!-- #endregion -->

```python id="JTmyht9d3Qr5"
# Mengubah teks menjadi huruf kecil
NoD['Nama Wisata'] = NoD['Nama Wisata'].str.lower()
```

<!-- #region id="3A7QzXET49FU" -->
### Encoding Fitur Kategorikal
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="siPz4qJqfI3H" outputId="aa4ccf37-6e56-49b3-95f6-c1d91d3ca813"
# Label Encoding
label_encoder_jenis_wisata = LabelEncoder()
label_encoder_kabupaten_kota = LabelEncoder()
label_encoder_nama_wisata = LabelEncoder()

NoD['Jenis Wisata Encoded'] = label_encoder_jenis_wisata.fit_transform(NoD['Jenis Wisata'])
NoD['Kabupaten/Kota Encoded'] = label_encoder_kabupaten_kota.fit_transform(NoD['Kabupaten/Kota'])
NoD['Nama Wisata Encoded'] = label_encoder_nama_wisata.fit_transform(NoD['Nama Wisata'])

label_mapping_jenis_wisata = dict(zip(label_encoder_jenis_wisata.classes_, label_encoder_jenis_wisata.transform(label_encoder_jenis_wisata.classes_)))
label_mapping_kabupaten_kota = dict(zip(label_encoder_kabupaten_kota.classes_, label_encoder_kabupaten_kota.transform(label_encoder_kabupaten_kota.classes_)))
label_mapping_nama_wisata = dict(zip(label_encoder_nama_wisata.classes_, label_encoder_nama_wisata.transform(label_encoder_nama_wisata.classes_)))

print("\nLabel Mapping Jenis Wisata:")
print(label_mapping_jenis_wisata)

print("\nLabel Mapping Kabupaten/Kota:")
print(label_mapping_kabupaten_kota)

print("\nLabel Mapping Nama Wisata:")
print(label_mapping_nama_wisata)
```

```python colab={"base_uri": "https://localhost:8080/"} id="uL8SUU045DHG" outputId="7652e918-cdf8-4327-b81d-9fe2651024c4"
# Lakukan one-hot encoding pada kolom "Jenis Wisata"
encoded_NoD = pd.get_dummies(NoD['Jenis Wisata'], prefix='Wisata')

# Gabungkan hasil encoding dengan DataFrame asli
NoD_encoded = pd.concat([NoD, encoded_NoD], axis=1)

print(NoD_encoded)
```

```python colab={"base_uri": "https://localhost:8080/"} id="A5qiZK7-8Q27" outputId="f01261d3-76d1-4aab-ba1d-7ebc89b18e93"
NoD.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ydSZUx3v_6g5" outputId="3200a8cb-1923-48c8-ba9d-a4a5e1b2e7ad"
encoded_NoD.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 982} id="D5SPtYyuB2GH" outputId="c71b7331-a975-42c0-debf-6dbd16e078e0"
# Gabungkan hasil encoding dengan DataFrame asli
NoD = pd.concat([NoD, encoded_NoD], axis=1)

NoD
```

<!-- #region id="AYNGu-_v40JQ" -->
### Menangani Data yang Tidak Sesuai atau Error
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aVh-fOZR8Sxl" outputId="421cf931-6f68-48f9-d25c-d3f81c13066a"
# Menampilkan isi data kolom Rating
print(NoD['Rating'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="LK7TfGpo90D5" outputId="acd78c94-99ed-42c8-829b-8bc43b068a8c"
# Menampilkan baris dengan nilai Rating di luar rentang yang diinginkan
outlier_ratings = NoD.loc[(NoD['Rating'] < 1) | (NoD['Rating'] > 5)]
print(outlier_ratings)
```

```python id="8jDRx3k_41z0"
# Menghapus baris dengan nilai yang tidak sesuai
NoD = NoD[NoD['Rating'].between(1, 5)]
```

```python colab={"base_uri": "https://localhost:8080/"} id="x4bqNIYC8a7y" outputId="4e1013af-bf31-49ce-a121-2257572f5224"
# Menampilkan isi data kolom Rating
print(NoD['Rating'])
```

<!-- #region id="OLwU6wt2wJAY" -->
# Save Dataset After Wrangling
<!-- #endregion -->

<!-- #region id="2OeJ-Ei5RShr" -->
Save Model h5
<!-- #endregion -->

```python id="NwIKTveDwIhn"
# menyimpan dataframe ke dalam file CSV
NoD.to_csv('NoD_clean.csv', index=False)
NoD_clean = pd.read_csv('./NoD_clean.csv')
```

<!-- #region id="t5pnofqE5c2I" -->
# Exploratory Data Analysis (EDA)
<!-- #endregion -->

```python colab={"referenced_widgets": ["1bd6e0b8aa5349a794d372843f78842d", "4bcac138e25544638af43dae6693e2dd", "335cee915a234e30a898633209bb2b11", "6e067e3dff1d49c78035ed1ff3981fa2", "14b8af4fc5224a609ca26928b088e8a9", "f4747611c67f45fa9e5b461b3d8a1875", "fa95ffdac75142f5b0401002798f164f", "6e853246626e49e689ceca6608de7f31", "2022641faf6048eb9906f6d9c63813fb", "fab75b6160554086ab3314c6569a48a1", "a3616af9ffa0485f94ab43053ca94d37", "f1e2b5e38ec44159954db06072ca809b", "66db866cae724f5ead03d7eddd2c5208", "f48b4661188e436a83af2bfea849bc5e", "9f020fb7c7a34948929191f6320daa24", "47a2a8909b8548d5a60465a7126d1822", "6aee4429215c49c19d98e14ec74b1440", "f96d3a9b809946fe891f17862304466c", "6a8ee5b288a740f09848796f036c6f9d", "d0c654ee311b4ad28308d7de17dfc5cb", "6cca83c73cf9490a83b227fc492fa29d", "4b4f7c38eaa040a98a975fc328651c73", "a1b6fdbbc72d45239f4d3b47fe28af59", "08bf7accfa82445daa10beca89ddbaf9", "1388d3b24a224da7b7d8a805c65d6dd3", "0301762f864d4e87a0dd920d0970d55f", "1531e66319674a90ba2f861ac2b16f91", "392f0e7021f94779b8b485145ed21fc6", "26f3c3b83af149aea022afb90a1c40b0", "806caa5c8d9549129d45ca14e7de28df", "ad1f9673c54d48cbab074a789b0f42f3", "412b4684d1ab45e4bc0f9f5c860173f8", "4412033cb5bc425787fae6da79732cff", "e8b3caae3b504f7f9b5cd5cabcac6c55", "ade0cde07e7b4d658b2f8c22749089cc", "d5de48afe36848cdb99fb1dada843e2a", "4d02ff974b8e4677b5085e370501f1e4", "19f7bd57c2204100b7015ea3a5aea690", "ce461485289d477ea5c2812b84df193f", "6f6f4f62160446f0b82774a5e71e8661", "a6ab2c2f934e49c98125467911119185", "1e38f60ca2614cd3aadce51bcd0e3e22", "f0f65859578f4e07a49596564e5990df", "122e5bd48be64967a13ce96272367f00"], "base_uri": "https://localhost:8080/", "height": 145} id="LE9iIasu6RD7" outputId="ebf43f6a-b086-4432-b091-37acca24ff32"
# Assuming NoD is your DataFrame
profile_NoD = ydata_profiling.ProfileReport(NoD)
profile_NoD.to_file("NoD Report.html")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 821} id="phi3geir6VLy" outputId="53fd30a0-ab74-451f-e93c-154b802d4550"
# Menampilkan hasil eksplorasi
profile_NoD
```

```python colab={"base_uri": "https://localhost:8080/"} id="pVjCuj7b7C7_" outputId="abac5ed2-f174-43e7-ef0c-a6376fa69491"
NoD.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 982} id="2Y95rEV8_LY3" outputId="2bfdaa72-6ac0-4167-f1c2-5cdbd3b62fd7"
NoD
```

<!-- #region id="MTgBm9BO_moC" -->
# Model Content Based Filltering (CBF)
<!-- #endregion -->

<!-- #region id="zz5eljLvxVCA" -->
## Model Rekomendasi Berdasarkan Prediksi Jenis Wisata
<!-- #endregion -->

<!-- #region id="DA4xcCmH28MS" -->
### Step 1: Hitung Jarak dengan Geodesic
<!-- #endregion -->

```python id="HroBZe_D2-fZ"
# Hitung jarak dari koordinat pengguna ke semua tempat wisata dalam dataset
def hitung_jarak(user_latitude, user_longitude, NoD_clean):
    jarak = []
    for index, tempat_wisata in NoD_clean.iterrows():
        jarak_tempat_wisata = geodesic((user_latitude, user_longitude), (tempat_wisata['Latitude'], tempat_wisata['Longitude'])).kilometers
        jarak.append(jarak_tempat_wisata)
    NoD_clean['Jarak'] = jarak
    return NoD_clean

# Koordinat pengguna (contoh)
user_latitude, user_longitude = -6.2088, 106.8456

# Hitung jarak
NoD_clean = hitung_jarak(user_latitude, user_longitude, NoD_clean)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} id="zBgzhynz3e0R" outputId="1a93dec8-24ae-4de3-f936-8267c0b7beb5"
NoD_clean
```

<!-- #region id="9vdJKbsuxZ7u" -->
### Step 2: Train Model
<!-- #endregion -->

```python id="nw5I2D4w1Nnp"
X = NoD_clean[['Kabupaten/Kota Encoded', 'Nama Wisata Encoded', 'Jarak', 'Wisata_Air', 'Wisata_Bukit', 'Wisata_Monumen', 'Wisata_Religi', 'Wisata_Taman']]
Y = NoD_clean['Jenis Wisata Encoded']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

```python id="addmFIcL1R_R"
# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membangun model MLP
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Output layer dengan jumlah neuron sesuai dengan jumlah kelas jenis wisata
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_jenis_wisata = create_model()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4TNKutu41v39" outputId="2d489b1b-6157-483a-f4e3-741c7b195522"
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Melatih model
history = model_jenis_wisata.fit(X_train, Y_train,
                                 validation_split=0.3,
                                 epochs=100,
                                 batch_size=32,
                                 callbacks=[early_stopping])
```

```python colab={"base_uri": "https://localhost:8080/"} id="2Ozc86Fr1x1j" outputId="2ad6b844-06b8-4588-be44-45286869d1e8"
# Evaluasi model
loss, accuracy = model_jenis_wisata.evaluate(X_test, Y_test)
print("Loss on test data:", loss)
print("Accuracy on test data:", accuracy)
```

<!-- #region id="AtfbfVaY72V3" -->
### Step 3: Saved Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 72} id="soTCEX4V74Vl" outputId="3f6db26f-8f1e-464b-c6d9-9341724a4feb"
model_jenis_wisata.save('Recommendation_Jenis_Wisata.h5')
print("Model saved as Recommendation_Jenis_Wisata.h5")
files.download('Recommendation_Jenis_Wisata.h5')
```

```python colab={"base_uri": "https://localhost:8080/"} id="7KrHps_Wp0L8" outputId="0d463361-abb6-4aa6-ac94-bbcb37d89f61"
model_jenis_wisata.export('mymodel')

import subprocess
command = [
    'tensorflowjs_converter',
    '--input_format', 'tf_saved_model',
    '--output_format','tfjs_graph_model',
    'mymodel',  # Input Keras model file
    'tfjs_model12'   # Output directory for the TensorFlow.js model
]
subprocess.run(command)
```

<!-- #region id="wYAkYsg7I4b0" -->
### Step 4: Testing Model
<!-- #endregion -->

<!-- #region id="YGPuZvnHxN_U" -->
#### Rekomendasi Berdasarkan Jenis Wisata
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rJxEqDceI3pD" outputId="bbe72d4d-832c-4701-ba8b-965c77f5a8f5"
# Load model yang sudah disimpan
model = load_model('Recommendation_Jenis_Wisata.h5')
model_jenis_wisata = model
```

```python id="t_rjH61CK7Ha"
from geopy.distance import geodesic
import numpy as np

def get_recommendation_by_distance_and_category(model_jenis_wisata, user_latitude, user_longitude, selected_category, NoD_clean):
    # Hitung jarak dari koordinat pengguna ke semua tempat wisata dalam dataset
    def hitung_jarak(user_latitude, user_longitude, NoD_clean):
        jarak = []
        for index, tempat_wisata in NoD_clean.iterrows():
            jarak_tempat_wisata = geodesic((user_latitude, user_longitude), (tempat_wisata['Latitude'], tempat_wisata['Longitude'])).kilometers
            jarak.append(jarak_tempat_wisata)
        NoD_clean['Jarak'] = jarak
        return NoD_clean

    # Hitung jarak
    NoD_clean = hitung_jarak(user_latitude, user_longitude, NoD_clean)

    # Filter tempat wisata berdasarkan jenis wisata yang dipilih pengguna
    filtered_data = NoD_clean[NoD_clean[selected_category] == 1]

    # Normalisasi data
    X_test = filtered_data[['Kabupaten/Kota Encoded', 'Nama Wisata Encoded', 'Jarak', 'Wisata_Air', 'Wisata_Bukit', 'Wisata_Monumen', 'Wisata_Religi', 'Wisata_Taman']]
    X_test_scaled = scaler.transform(X_test)

    # Prediksi jenis wisata
    predicted_category = np.argmax(model_jenis_wisata.predict(X_test_scaled), axis=1)

    # Gabungkan hasil prediksi dengan dataset
    filtered_data['Predicted Category'] = predicted_category

    # Sorting berdasarkan jarak terdekat
    recommended_places = filtered_data.sort_values(by='Jarak').head(15)

    return recommended_places

# Label Mapping Jenis Wisata:
# {'Air': 0, 'Bukit': 1, 'Monumen': 2, 'Religi': 3, 'Taman': 4}
```

```python colab={"base_uri": "https://localhost:8080/"} id="A45xt_hOGBxb" outputId="1b410762-9493-4a24-bf4a-9294256a848e"
# Contoh penggunaan 1
selected_category = 'Wisata_Taman'
user_latitude, user_longitude = -8.123566881348339, 115.06560150999103  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance_and_category(model_jenis_wisata, user_latitude, user_longitude, selected_category, NoD_clean)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

```python colab={"base_uri": "https://localhost:8080/"} id="eiWfBXUWcV6P" outputId="1a11cf97-a502-4961-f028-dd6a205326fe"
# Contoh penggunaan 2
selected_category = 'Wisata_Air'
user_latitude, user_longitude = -8.123566881348339, 115.06560150999103  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance_and_category(model_jenis_wisata, user_latitude, user_longitude, selected_category, NoD)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z8hp0FJHcgcy" outputId="44bb4297-ccd0-4f15-9542-a64d303b2102"
# Contoh penggunaan 3
selected_category = 'Wisata_Religi'
user_latitude, user_longitude = -8.123566881348339, 115.06560150999103  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance_and_category(model_jenis_wisata, user_latitude, user_longitude, selected_category, NoD)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

<!-- #region id="fZ_-11drxdGR" -->
#### Rekomendasi Berdasarkan Jarak
<!-- #endregion -->

```python id="G9WBr0BjzZlp"
def get_recommendation_by_distance(model_jenis_wisata, user_latitude, user_longitude, NoD_clean):
    # Hitung jarak dari koordinat pengguna ke semua tempat wisata dalam dataset
    def hitung_jarak(user_latitude, user_longitude, NoD_clean):
        jarak = []
        for index, tempat_wisata in NoD_clean.iterrows():
            jarak_tempat_wisata = geodesic((user_latitude, user_longitude), (tempat_wisata['Latitude'], tempat_wisata['Longitude'])).kilometers
            jarak.append(jarak_tempat_wisata)
        NoD_clean['Jarak'] = jarak
        return NoD_clean

    # Hitung jarak
    NoD_clean = hitung_jarak(user_latitude, user_longitude, NoD_clean)

    # Normalisasi data
    X_test = NoD_clean[['Kabupaten/Kota Encoded', 'Nama Wisata Encoded', 'Jarak', 'Wisata_Air', 'Wisata_Bukit', 'Wisata_Monumen', 'Wisata_Religi', 'Wisata_Taman']]
    X_test_scaled = scaler.transform(X_test)

    # Prediksi jenis wisata
    predicted = np.argmax(model_jenis_wisata.predict(X_test_scaled), axis=1)

    # Gabungkan hasil prediksi dengan dataset
    NoD_clean['Predicted Category'] = predicted

    # Sorting berdasarkan jarak terdekat
    recommended_places = NoD_clean.sort_values(by='Jarak').head(15)

    return recommended_places

# Label Mapping Jenis Wisata:
# {'Air': 0, 'Bukit': 1, 'Monumen': 2, 'Religi': 3, 'Taman': 4}
```

```python colab={"base_uri": "https://localhost:8080/"} id="IQWbn_g-GG2a" outputId="d4917330-b2d9-4f0d-b800-01d23afed9f2"
# Contoh penggunaan 1
user_latitude, user_longitude = -8.123566881348339, 115.06560150999103  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance(model_jenis_wisata, user_latitude, user_longitude, NoD_clean)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

```python colab={"base_uri": "https://localhost:8080/"} id="_YfE3RgXxqM2" outputId="465730fa-b4e7-48b7-ed43-5b973656578b"
# Contoh penggunaan 2
user_latitude, user_longitude = -6.472807961145061, 106.8630118908314  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance(model_jenis_wisata, user_latitude, user_longitude, NoD)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

<!-- #region id="yOVDRqdExm6R" -->
#### Rekomendasi Berdasarkan Reviews
<!-- #endregion -->

```python id="s7c3RFpSFYH5"
def get_recommendation_by_distance_and_reviews(model_jenis_wisata, user_latitude, user_longitude, NoD_clean):
    # Hitung jarak dari koordinat pengguna ke semua tempat wisata dalam dataset
    def hitung_jarak(user_latitude, user_longitude, NoD_clean):
        jarak = []
        for index, tempat_wisata in NoD_clean.iterrows():
            jarak_tempat_wisata = geodesic((user_latitude, user_longitude), (tempat_wisata['Latitude'], tempat_wisata['Longitude'])).kilometers
            jarak.append(jarak_tempat_wisata)
        NoD_clean['Jarak'] = jarak
        return NoD_clean

    # Hitung jarak
    NoD_clean = hitung_jarak(user_latitude, user_longitude, NoD_clean)

    # Sorting berdasarkan jarak terdekat
    recommended_places = NoD_clean.sort_values(by='Jarak').head(15)

    # Urutkan berdasarkan banyaknya ulasan (reviews) secara menurun
    recommended_places = recommended_places.sort_values(by='Reviews', ascending=False)

    return recommended_places
```

```python colab={"base_uri": "https://localhost:8080/"} id="hLgjTSH0FYBF" outputId="3810568e-590c-4ee1-a6cb-10353fc7215b"
# Contoh penggunaan 1
user_latitude, user_longitude = -8.652998964078941, 115.21385652533104  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan banyaknya ulasan (reviews)
recommended_places = get_recommendation_by_distance_and_reviews(model_jenis_wisata, user_latitude, user_longitude, NoD_clean)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak']])
```

```python colab={"base_uri": "https://localhost:8080/"} id="DRCwyJPSGgiP" outputId="430cd875-6a48-4df1-f6df-85f8304a1ccd"
# Contoh penggunaan 2
user_latitude, user_longitude = -6.472807961145061, 106.8630118908314  # Koordinat pengguna (contoh)

# Mendapatkan rekomendasi berdasarkan jarak dan jenis wisata pilihan pengguna
recommended_places = get_recommendation_by_distance_and_reviews(model_jenis_wisata, user_latitude, user_longitude, NoD)

# Menampilkan hasil rekomendasi
print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
```

<!-- #region id="KnPiIDHTTYRh" -->
#### Testing by Choose the Recommendation Model
<!-- #endregion -->

```python id="2QnD2iGWTOAX"
def switch_case_test(test_case, model_jenis_wisata, NoD_clean):
    if test_case == 1:
        user_latitude = float(input("Masukkan latitude pengguna: "))
        user_longitude = float(input("Masukkan longitude pengguna: "))
        selected_category = input("Masukkan jenis wisata yang dipilih: ")
        recommended_places = get_recommendation_by_distance_and_category(model_jenis_wisata, user_latitude, user_longitude, selected_category, NoD_clean)
    elif test_case == 2:
        user_latitude = float(input("Masukkan latitude pengguna: "))
        user_longitude = float(input("Masukkan longitude pengguna: "))
        recommended_places = get_recommendation_by_distance(model_jenis_wisata, user_latitude, user_longitude, NoD_clean)
    elif test_case == 3:
        user_latitude = float(input("Masukkan latitude pengguna: "))
        user_longitude = float(input("Masukkan longitude pengguna: "))
        recommended_places = get_recommendation_by_distance_and_reviews(model_jenis_wisata, user_latitude, user_longitude, NoD_clean)
    else:
        print("Invalid test case number. Please choose 1, 2, or 3.")
        return None
    return recommended_places
```

```python id="bVx37yzwTNWu" colab={"base_uri": "https://localhost:8080/"} outputId="72595706-9f63-4c4d-dd8d-e4e24789a8cd"
test_case = int(input("Masukkan nomor test case (1/2/3): "))  # Pilih nomor test case

if isinstance(NoD_clean, pd.DataFrame):
    recommended_places = switch_case_test(test_case, model_jenis_wisata, NoD_clean)
    if recommended_places is not None:
        print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
else:
    print("NoD_clean bukan DataFrame yang valid. Pastikan data telah dimuat dengan benar.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="SQ8VpIhUTsnO" outputId="cec90914-6493-45ec-cbbf-aa0b19645c38"
test_case = int(input("Masukkan nomor test case (1/2/3): "))  # Pilih nomor test case

if isinstance(NoD_clean, pd.DataFrame):
    recommended_places = switch_case_test(test_case, model_jenis_wisata, NoD_clean)
    if recommended_places is not None:
        print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
else:
    print("NoD_clean bukan DataFrame yang valid. Pastikan data telah dimuat dengan benar.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z5_ycKidT1X3" outputId="4d214888-37ce-4eca-91e1-a9a7b841a3cd"
test_case = int(input("Masukkan nomor test case (1/2/3): "))  # Pilih nomor test case

if isinstance(NoD_clean, pd.DataFrame):
    recommended_places = switch_case_test(test_case, model_jenis_wisata, NoD_clean)
    if recommended_places is not None:
        print(recommended_places[['Kabupaten/Kota', 'Nama Wisata', 'Jenis Wisata', 'Rating', 'Reviews', 'Jarak', 'Predicted Category']])
else:
    print("NoD_clean bukan DataFrame yang valid. Pastikan data telah dimuat dengan benar.")
```
