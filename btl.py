import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
saved_model = tf.keras.models.load_model("sang1.h5")
st.header("Nhận diện chữ số viết tay nhóm 12!!!!")



f = st.file_uploader("Upload file")
if f:
    if f.name[-3:] in ('jpg', 'png', 'JPG', 'PNG'):
        st.write("Số bạn vừa gửi là ")
        st.image(f)
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # # imgin = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
    # st.write("Wating...")
    # print(opencv_image)
    img = cv2.cvtColor(opencv_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (28,28))
    x = img_to_array(img) 
    x = tf.image.rgb_to_grayscale(x) 
    
    images=np.reshape(x,(28,28)) 
    a=images
    images = np.expand_dims(images, axis=0)
    result = saved_model.predict(images)
    y_pre = [np.argmax(element) for element in result]
    st.write("Dự đoán số bằng mạng CNN là :  ",y_pre[0])