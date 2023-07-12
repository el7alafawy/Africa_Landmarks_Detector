import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
    
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_'
# model_url = 'on_device_vision_classifier_landmarks_classifier_africa_V1_1'

Africalabels = './Labels/Africa_Label_Map.csv'
df = pd.read_csv(Africalabels)
Africalabels = dict(zip(df.id, df.name))

Asialabels = './Labels/Asia_Label_Map.csv'
df = pd.read_csv(Asialabels)
Asialabels = dict(zip(df.id, df.name))

Australialabels = './Labels/Australia_Label_Map.csv'
df = pd.read_csv(Australialabels)
Australialabels = dict(zip(df.id, df.name))

Europelabels = './Labels/Europe_Label_Map.csv'
df = pd.read_csv(Europelabels)
Europelabels = dict(zip(df.id, df.name))

NorthAmericalabels = './Labels/NorthAmerica_Label_Map.csv'
df = pd.read_csv(NorthAmericalabels)
NorthAmericalabels = dict(zip(df.id, df.name))

SouthAmericalabels = './Labels/SouthAmerica_Label_Map.csv'
df = pd.read_csv(SouthAmericalabels)
SouthAmericalabels = dict(zip(df.id, df.name))

if 'model' not in st.session_state:
    st.session_state.model = 'None'

def image_processing(image,model,labels):
    img_shape = (321, 321)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model, input_shape=img_shape + (3,), output_key="predictions:logits")])
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    result = classifier.predict(img)
    return labels[np.argmax(result)],img1

def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address,location.latitude, location.longitude

def run(model):
    st. set_page_config(layout="wide")
    st.title("Landmarks Guide")
    img = PIL.Image.open('logo.jpeg')
    img = img.resize((256,256))
    st.image(img)
    col1, col2, col3,col4, col5, col6 = st.columns(6)
    with col1:
        st.header("Asia 🍣")
        img = PIL.Image.open('./images/Asia.jpg')
        img = img.resize((300,450))
        st.image(img)
        if st.button('Choose'):
            model = "Asia"
            st.session_state.model = model
    with col2:
        st.header("Europe 🥐")
        img = PIL.Image.open('./images/Europe.jpg')
        img = img.resize((300,450))
        st.image(img)
        if st.button('Choose',1):
            model = "Europe"
            st.session_state.model = model
    with col3:
        st.header("Africa 🥙")
        img = PIL.Image.open('./images/Africa.jpg')
        img = img.resize((300,450))
        st.image(img)
        if st.button('Choose',2):
            model = "Africa"
            st.session_state.model = model
    with col4:
        st.header("North America 🥩")
        img = PIL.Image.open('./images/NorthAmerica.jpg')
        img = img.resize((300,390))
        st.image(img)
        if st.button('Choose',3):
            model = "NorthAmerica"
            st.session_state.model = model
    with col5:
        st.header("South America 🌮")
        img = PIL.Image.open('./images/SouthAmerica.jpg')
        img = img.resize((300,390))
        st.image(img)
        if st.button('Choose',4):
            model = "SouthAmerica"
            st.session_state.model = model
    with col6:
        st.header("Australia 🧈")
        img = PIL.Image.open('./images/Australia.jpg')
        img = img.resize((300,450))
        st.image(img)
        if st.button('Choose',5):
            model = "Australia"
            st.session_state.model = model
    if model != "None":
        st.subheader("Searching landmarks in " + model)
        img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
        cam_chk = st.checkbox("Use camera")
        if cam_chk:
            cam_file = st.camera_input("Fresh from camera")
        else:
            cam_file = None
        if cam_file is not None or img_file is not None:
            if cam_file is not None :
                ch_img = cam_file
            else:
                ch_img = img_file
            save_image_path = './Uploaded_Images/' + ch_img.name
            
            with open(save_image_path, "wb") as f:
                f.write(ch_img.getbuffer())

            #  Selecting Model & Labels
            if model == "Australia":
                ChoosenModelURl = model_url +"oceania_antarctica_V1/1"
                Labels = Australialabels
            elif model == "NorthAmerica":
                ChoosenModelURl = model_url +"north_america_V1/1"
                Labels = NorthAmericalabels
            elif model == "SouthAmerica":
                ChoosenModelURl = model_url +"south_america_V1/1"
                Labels = SouthAmericalabels
            elif model == "Africa":
                ChoosenModelURl = model_url +"africa_V1/1"
                Labels = Africalabels
            elif model == "Europe":
                ChoosenModelURl = model_url +"europe_V1/1"
                Labels = Europelabels
            elif model == "Asia":
                ChoosenModelURl = model_url +"asia_V1/1"
                Labels = Asialabels
            else :
                ChoosenModelURl = ""
                Labels = None
            prediction,image = image_processing(save_image_path,ChoosenModelURl,Labels)
            st.image(image)
            st.header("📍 **Predicted Landmark is: " + prediction + '**')
            try:
                address, latitude, longitude = get_map(prediction)
                st.success('Address: '+address )
                loc_dict = {'Latitude':latitude,'Longitude':longitude}
                st.subheader('✅ **Latitude & Longitude of '+prediction+'**')
                st.json(loc_dict)
                data = [[latitude,longitude]]
                df = pd.DataFrame(data, columns=['lat', 'lon'])
                st.subheader('✅ **'+prediction +' on the Map**'+'🗺️')
                st.map(df)
            except Exception as e:
                st.warning("No address found!!")
    else:
        st.write("Please choose any continent!")
run(st.session_state.model)
