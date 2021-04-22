import re
import sys
from streamlit.cli import main
import streamlit as st 
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from pyngrok import ngrok

st.title("Image Classifier using Machine Learning")
st.text('Upload the Image')

model = pickle.load(open('image_model.p', 'rb'))
uploaded_file = st.file_uploader("Choose an image...", type='jpg')
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded image')
    
    if st.button('Predict'):
        categories = ["alien", "predator"]
        st.write('Result....')
        flat_image = []
        image = np.array(img)
        image_resized = resize(image, (150, 150, 3))
        flat_image.append(image_resized.flatten())
        flat_image = np.array(flat_image)
        plt.imshow(image_resized)
        output = model.predict(flat_image)
        output = categories[output[0]]
        st.title(f'Predicted output: {output}')
        

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
        
