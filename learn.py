#Importing libraries
import numpy as np  
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import streamlit as st  

#####################################
if "step" not in st.session_state:
    st.session_state.step = 1

if "info" not in st.session_state:
    st.session_state.info = {}



def go_topage3():
    st.session_state.step = 3

def go_topage2():
    st.session_state.step = 2


def go_topage1():
    st.session_state.step=1


if st.session_state.step == 1:
    st.title("Welcome to KejaPredict.AI")
    st.divider()
    st.header("Built by Kenyan for Kenyans")

    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Get to know about KejaPredict.AI",on_click=go_topage2,use_container_width=True)
        
    with col2:
            st.button("Predict house price", on_click=go_topage3,use_container_width=True)




    

    

elif st.session_state.step == 2:
    st.header("What is KejaPredict.AI?")
    st.write("KejaPredict.AI is an intelligent real estate pricing platform that uses machine learning to predict both property sale prices and rental values across Kenya. By analyzing critical factors such as location, property size, and amenities, the platform delivers fast and reliable insights to tenants, landlords, buyers, and investors. KejaPredict.AI empowers users withdata-driven decisions, bringing transparency and efficiency to the Kenyan real estate market.")
    st.divider()
    st.subheader("How does it work?")
    st.write("KejaPredict.AI utilizes a machine learning model trained on a comprehensive dataset of Kenyan real estate transactions. The model considers various features such as location, property size, and amenities to predict accurate sale prices and rental values. Users can input specific property details to receive instant predictions, enabling informed decisions in the dynamic Kenyan real estate market.")
    st.divider()
    st.button("Go back to page 1", on_click=go_topage1)

elif st.session_state.step == 3:
    st.header("Predict House Price")

    df = pd.read_csv("housing.csv")
    df = df.drop("ocean_proximity", axis =1)
    df = df.fillna(0)
    #data splitting
    features =  df.drop("median_house_value",axis = 1)
    target = df["median_house_value"]

    X_max = features.max()
    y_max = target.max()

    X = features/X_max
    y = target/y_max

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#making model
    model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=[8,]),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

#compling the model
    model.compile(optimizer = 'adam',loss='mean_squared_error',metrics = ['mean_absolute_error'])

#training the model
    model.fit(X_train,y_train,epochs = 100, batch_size = 32, validation_split = 0.2)



    
#evaluate the model
    test_loss, test_mae = model.evaluate(X,y)
    print("Test Loss:", test_loss)

#making predictions
    predictions = model.predict(X_test)

    longitude = st.number_input("Enter longitude (e.g., -122.23): ")
    latitude = st.number_input("Enter latitude (e.g., 37.88): ")
    housing_median_age = st.number_input("Enter house age (e.g., 41.0): ")
    total_rooms = st.number_input("Enter total rooms (e.g., 880.0): ")
    total_bedrooms = st.number_input("Enter total bedrooms (e.g., 129.0): ")
    population = st.number_input("Enter population (e.g., 322.0): ")
    households = st.number_input("Enter households (e.g., 126.0): ")
    median_income = st.number_input("Enter median income (e.g., 8.3252): ")

    your_search = np.array([[
  longitude, latitude, housing_median_age, total_rooms,
   total_bedrooms, population, households, median_income
]])

# convert to numpy
    search = your_search / X_max.values

    prediction = model.predict(search)

    final_price = prediction[0][0] * y_max

    print(f"Predicted house price: {final_price}")

    st.button("Submit")
    if longitude and latitude and housing_median_age and total_rooms and total_bedrooms and population and households and median_income:
        st.success("Prediction successful!")
        st.divider()
        st.write(f"Predicted house price is around: {final_price}")
        st.warning("Note: This is an estimate based on the input data and may not reflect the actual market value.")
    else:
        st.error("Please fill in all the fields to get a prediction.")



