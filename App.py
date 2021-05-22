
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder




import base64



st.set_page_config(  # Alternate names: setup_page, page, layout
layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
page_title= "ArztAI",  # String or None. Strings get appended with "• Streamlit". 
page_icon= "favicon.png", # String, anything supported by st.image, or None.
)

main_bg = "Images/BACK2.png"
main_bg_ext = "Images/BACK2.png"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)





col1, col2, col3 = st.beta_columns([1,2,1])
col1.write("")
col2.image("logo.png", width = 350)
col3.write("")

st.sidebar.image("ARZT-AI.png", width = 300)



menu = ["Home", "Login"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
  st.write(""" <p style=" color:#FFD700; font-size: 30px; font-weight:bold"> We're ArztAI, an innovative digital health platform.</p>
  """, unsafe_allow_html=True)

  st.write("")

  st.write("""  <p style=" font-size: 15px; font-weight:normal; font-family:verdana"> ArztAI is engaged in the implementation of artificial intelligence technologies in medicine and provide AI-based health solutions. We're transforming the diagnostic system, improving the quality
of medical services in general, and reducing costs. Our platform shows excellent results - the accuracy of detecting pathologies by ultrasound and MRI exceeds 92%.</p>
  """, unsafe_allow_html=True)

  st.write("")

  col1, col2 = st.beta_columns(2)
  col1.image("Covid_Normal_Pneumonia.png", width = 600)
  col2.image("Tumor_Normal.png", width = 600)
  
  st.write("")

  st.write("""  <p style=" font-size: 15px; font-weight:normal; font-family:verdana""> Platform with integrated Artificial intelligence enables detection of diseases even at an early stage. ArztAI helps doctors correctly interpret the necessary information and notice every detail.Artificial intelligence searches medical images for patterns characteristic of diseases (tumors, pneumonia, coronavirus, and so on), 
and highlights the smallest details that a person would not pay attention to.</p>
  """, unsafe_allow_html=True)

  st.write("")

  st.write(""" <p style=" color:#FFD700; font-size: 30px; font-weight:bold"> Three steps to be health ...</p>
  """, unsafe_allow_html=True)

  col1, col2, col3 = st.beta_columns(3)
  col1.image("sistem.png", width = 400)
  col2.image("sistem2.png", width = 400)
  col3.image("sistem3.png", width = 400)

  st.write("")


  st.write(""" <p style=" color:#33B5FF; font-size: 30px; font-weight:bold"> Cooperation </p>
  """, unsafe_allow_html=True)

  st.write("")

  st.write("""  <p style=" font-size: 15px; font-weight:normal; font-family:verdana""> This app aimed just for doctors and provides only information, is not medical or treatment advice. Interpretation of the results and the final decision are must be given by the doctor. 
  We are always open to new cooperation, and if you are interested in cooperation you can use the contact information below. 
  </p>
  """, unsafe_allow_html=True)

  st.image("contact.png", width = 250)


elif choice == "Login":
  username = st.sidebar.text_input("User Name")
  password = st.sidebar.text_input("Password", type = "password")
  if st.sidebar.checkbox("Login"):
    if password == "537382" and username == "Admin":
      st.success("Logged in as {}".format(username))
      st.write(""" <p style=" color:#FFD700; font-size: 30px; font-weight:bold"> Model Selection </p>
  """, unsafe_allow_html=True)
      task = st.selectbox("Task",["COVID19 and Pneumonia Detection", "Brain Tumor Detection", "Heart Failure Risk Prediction"])
      if task == "COVID19 and Pneumonia Detection":
        st.subheader("Chest X-ray")
        @st.cache(allow_output_mutation = True)
        def load_model():
          model = tf.keras.models.load_model("model_new_COVID19.h5")
          return model
          
        model = load_model()
        file = st.file_uploader("Please upload an Chest X-ray image", type = ["jpeg","jpg","png"])
        
        def import_and_predict(image_data, model):
          image_data = image_data.convert('RGB')
          image_data = np.array(image_data)
          img = cv2.resize(image_data,(224,224))     # resize image to match model's expected sizing
          img = img.reshape([1,224,224,3])
          img = preprocess_input(img) #
          prediction = model.predict(img)
          return prediction
          
        if file is None:
          st.text("Please upload an image file")
        else:
          image = Image.open(file)
          st.image(image,  use_column_width = True)
          prediction = import_and_predict(image, model)
          class_names = ["COVID19", "Normal", "Pneumonia"]
          string = "Result is: " + class_names[np.argmax(prediction)]
          if class_names[np.argmax(prediction)] == "Normal":
            st.success("Result is: " + class_names[np.argmax(prediction)])
          else:
            st.error("Result is: " + class_names[np.argmax(prediction)])
        
      elif task == "Brain Tumor Detection":
        st.subheader("Brain MRI image")
        @st.cache(allow_output_mutation = True)
        def load_model():
          model = tf.keras.models.load_model("model_brain_tumor.h5")
          return model
          
        model = load_model()
        file = st.file_uploader("Please upload an Brain MRI image", type = ["jpeg","jpg","png"])
        
        def import_and_predict(image_data, model):
          image_data = image_data.convert('RGB')
          image_data = np.array(image_data)
          img = cv2.resize(image_data,(224,224))     # resize image to match model's expected sizing
          img = img.reshape([1,224,224,3])
          img = preprocess_input(img) #
          prediction = model.predict(img)
          return prediction
          
        if file is None:
          st.text("Please upload an image file")
        else:
          image = Image.open(file)
          st.image(image,  use_column_width = True)
          prediction = import_and_predict(image, model)
          class_names = ["Normal", "Tumor"]
          string = "Result is: " +  class_names[np.argmax(prediction)]
          if class_names[np.argmax(prediction)] == "Normal":
            st.success("Result is : "  +  class_names[np.argmax(prediction)])
          else:
            st.error("Result is : "  + class_names[np.argmax(prediction)])
          

          
          

      
      elif task == "Heart Failure Risk Prediction":
        st.subheader("Select Model Inputs")
        def input_features():
          age = st.slider("age", 18, 100, 45 )
          anaemia = st.selectbox("anaemia",("No","Yes"))
          creatinine_phosphokinase = st.text_input("creatinine_phosphokinase","800")
          diabetes = st.selectbox("diabetes",("No","Yes"))
          ejection_fraction = st.text_input("ejection_fraction","45")
          high_blood_pressure = st.selectbox("high_blood_pressure", ("No","Yes"))
          platelets = st.text_input("platelets","800")
          serum_creatinine = st.text_input("serum_creatinine", "0.6")
          serum_sodium = st.text_input("serum_sodium", "140")
          sex = st.selectbox("sex", ("Male","Female"))
          smoking = st.selectbox("smoking", ("No","Yes"))
          time = st.slider("time", 1,365,20)
          data = {"age": age,
                  "anaemia":anaemia,
                  "creatinine_phosphokinase":creatinine_phosphokinase,
                  "diabetes":diabetes,
                  "ejection_fraction":ejection_fraction,
                  "high_blood_pressure":high_blood_pressure,
                  "platelets":platelets,
                  "serum_creatinine":serum_creatinine,
                  "serum_sodium":serum_sodium,
                  "sex":sex,
                  "smoking":smoking,
                  "time":time}
        
          features = pd.DataFrame(data, index =[0])
          return features
    
        input_df = input_features()

        df_raw = pd.read_csv("heart_failure.csv")
        data = df_raw.drop(["DEATH_EVENT", "Unnamed: 0"], axis = 1)
        df = pd.concat([input_df,data],axis=0)
        
        # Encoding of ordinal features
        encode = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
        
        for col in encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])


        df = df[:1]

        st.subheader("User Input Features")
        st.write(df)

        # Reads in saved classification model
        rf = joblib.load("Heart_Faliure_model")
        # Apply model to make predictions
        prediction = rf.predict(df)
        prediction_proba = rf.predict_proba(df)

        st.subheader("Prediction")
        patient_types = np.array([0,1])
        st.write(patient_types[prediction])
        st.subheader("Prediction Probability")
        st.write(prediction_proba)
        if patient_types[prediction] == 1:
          st.error("At the risk group, check additional tests.")
        else:
          st.success("Patient not in the risk group")


    else:
      st.warning("Incorrect Username/Password") 
