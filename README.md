# **NoD (Near of Destination) Travel Destination Recommendation System**
Team ID : `C241-PS414`

## Description
> Near of Destination (NoD) is a recommendation system designed to help users find the best tourist attractions close to their location. The project utilizes various techniques in data wrangling, data exploration, and machine learning to provide accurate and relevant recommendations based on user preferences.
---
## Table of Contents
- [Installation and Import Libraries](#installation)
- [Data Wrangling](#data-wrangling)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Recommendation Methods](#recommendation-methods)
- [Save Model](#save-model)
- [Testing and Evaluation](#testing-and-evaluation)
  
## 1. Installation and Import Libraries
  - **Instalasi Libraries**:
    ```
    pip install numpy
    pip install scikit-learn
    pip install surprise
    pip install geopy
    pip install pandas
    pip install tensorflow
    pip install tensorflowjs
    pip install ydata-profiling
    pip install folium
    ```
  - **Import Libraries**:
    ```
    import pandas as pd
    import numpy as np
    import sklearn
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from geopy.distance import geodesic
    import ydata_profiling
    import os
    import shutil
    from google.colab import drive
    from google.colab import files
    import tensorflowjs as tfjs
    import warnings
    warnings.filterwarnings("ignore")
    import folium
    from folium.plugins import MarkerCluster
    ```

## 2. Data Wrangling
  - **Gathering Data**:
    - Accessing datasets from Google Drive.
    - Load the dataset using pd.read_csv.
  - **Assessing Data**:
    - Checking data structure, data quality (missing values, duplication, outliers).
    - Descriptive statistical analysis (histogram, boxplot).
  - **Cleaning Data**:
    - Handling missing values.
    - Removing duplication.
    - Correct data inconsistencies.
    - Encoding categorical features.
    - One-hot encoding.
    - Handle data that does not match or error.

## 3. Exploratory Data Analysis (EDA)
  - Profiling and data reporting using ydata_profiling.
  - Geospatial analysis using folium.

## 4. Data Modeling
  - Content Based Filtering (CBF):
    - Step 1: **Calculate Distance with Geodesic**:
      - Calculates the distance from the user's coordinates to all tourist attractions.
    - Step 2: **Train Model**:
      - Divide the dataset into training and testing.
      - Normalize the data.
      - Build MLP (Multilayer Perceptron) model.
      - Training the model.
      - Evaluate the model.
    - Step 3: **Save Model**:
      - Saving the model in .h5 format.
      - Convert the model to TensorFlow.js format.
    - Step 4: **Testing Model**:
      - Get recommendations based on category and distance.

## 5. Recommendation
  - **Based on Tourism Type**:
    - Calculates the distance and recommends based on the user's selected tour category.
    <img src="https://github.com/Near-of-Destination-NoD-C241-PS414/machine-learning/blob/main/Dokumentasi/Result%20of%20Recommendation%20by%20Jenis%20Wisata.png" alt="Result of Recommendaation by Type of Tourism" width="800"/>
  - **Based on Distance**:
    - Calculates distance and recommends nearby tourist attractions.
    <img src="https://github.com/Near-of-Destination-NoD-C241-PS414/machine-learning/blob/main/Dokumentasi/Result%20of%20Recommendaation%20by%20Jarak.png" alt="Result of Recommendaation by Distance" width="800"/>
  - **Based on Reviews**:
    - Calculates distance and sorts based on the number of reviews.
    <img src="https://github.com/Near-of-Destination-NoD-C241-PS414/machine-learning/blob/main/Dokumentasi/Result%20of%20Recommendation%20by%20Reviews.png" alt="Result of Recommendaation by Reviews" width="800"/>

## 6. Save Model
  ```
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

## 7. Testing and Evaluation
  - Testing the model with various user scenarios to get recommendations.

# Author:
1. (ML) M298D4KY2810- David Mario Yohanes Samosir - Ganesha University of Education
2. (ML) M298D4KY2631- Komang Wibisana - Ganesha University of Education
3. (ML) M298D4KY3353- Putu Gede Dimas Witjaksana - University of Ganesh Education
