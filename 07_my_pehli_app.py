import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2


st.title(":desert_island: Welcome to my first app :desert_island:")

st.header("A web app to convert your pic into pencil sketch")
st.subheader(":coffee: Enjoy this app and credits to #Codanics channel :coffee:")

def dodgeV2(x,y):
    return cv2.divide(x,255-y,scale=256)

def pen_sketch(inp_image):
    img_gray = cv2.cvtColor(inp_image,cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21,21),sigmaX=0,sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    return(final_img)


image = st.sidebar.file_uploader("Upload your image", type=['jpeg','jpg','png'])
if image is None:
    st.write("Please upload an image to be processed")
else:
    input_image = Image.open(image)
    final_sketch = pen_sketch(np.array(input_image))
    st.write("**Input Image**")
    st.image(input_image, use_column_width=False)
    st.write("**Pencil Sketch Image**")
    st.image(final_sketch,use_column_width=False)

