import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
import os
import streamlit as st

#function to convert an image to array
def convert_image_to_array(image_dir):
      try:
        image = cv2.imread(image_dir)
        
        if image is not None :
          image = cv2.resize(image, (256,256))
          return img_to_array(image)
        else:
          return np.array([])
      except Exception as e:
         print(f"Error : {e}")
         return None 

def convert_img(train_dir_path,Classes):
    dir = train_dir_path
    root_dir = os.listdir(dir)
    image_list , label_list = [] , []
    binary_labels = [0,1]
    temp = -1
    #Reading and converting image to numpy array
    for directory in root_dir:
        images1 = os.listdir(f"{dir}\{directory}")
        st.info("Images present in "+ str(directory)+ ": " + str(len(images1)))
        temp += 1
        for files in images1:
            image_path = f"{dir}\{directory}\{files}"
            image_list.append(convert_image_to_array(image_path))
            label_list.append(binary_labels[temp])


    label_list = np.array(label_list)
    label_list.shape
    import train_main
    st.warning("Select Epochs: ")
    epochs = st.sidebar.slider("Epochs: ",0,100,step=10)
    if(st.checkbox("Epochs Selected? ")):
      train_main.train(train_dir_path, image_list, label_list, Classes,epochs)
      train_main.deploy(Classes,train_dir_path)
    
   


