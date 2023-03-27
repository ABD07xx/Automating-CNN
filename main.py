import streamlit as st
import os
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time

st.markdown("""
            <div style='background-color: #cc00ff; padding: 10px'>
                <h1 style='text-align: center'>Automating CNN</h1>
            </div>
            """, unsafe_allow_html=True)


train_dir = st.file_uploader('Upload the training dataset', type='zip')
#if st.button('Info'):
if train_dir:
    # Set the path to the dataset folder
    dataset_folder = train_dir.name
    if train_dir is not None:
        # Extract the dataset from the zip file
        with zipfile.ZipFile(train_dir, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
            
        # Set the path to the training directory
        train_dir_path = os.path.join(dataset_folder, os.listdir(dataset_folder)[0])
        
        # Display the path to the training directory
        st.info(f'The path to the training directory is {train_dir_path}')
        Classes = len(os.listdir(train_dir_path))
        all_labels  = os.listdir(train_dir_path)
        st.info('Number of classes present = '+str(Classes))
        st.write("Classes present are: ",all_labels)

        import functions
        functions.convert_img(train_dir_path,Classes)
            












