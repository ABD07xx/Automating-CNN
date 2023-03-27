import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

def deploy_main(Classes,train_dir_path):
    #Loading the Model
    model = load_model(train_dir_path+".h5")

    #Name of Classes
    CLASS_NAMES = Classes

    #Setting Title of App
    st.title("Deployment")
    st.markdown("Upload an image")

    #Uploading the dog image
    cell_image = st.file_uploader("Choose an image...", type= ['png', 'jpg','jpeg'])


    if cell_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(cell_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (64,64))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,64,64,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(result)
