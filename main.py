#! Python3
# Predicts the emotion of an image containing a face
# resource : https://www.analyticsvidhya.com/blog/2021/10/machine-learning-model-deployment-using-streamlit/
# for deployment: https://medium.com/@pokepim/deploying-streamlit-app-to-ec2-instance-7a7edeffbb54
# https://aws.amazon.com/premiumsupport/knowledge-center/ec2-ppk-pem-conversion/
#
import send2trash
import helper
from helper import *
import streamlit as st
import os
import matplotlib.pyplot as plt

st.title('Emotions Classifier')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded_images',uploaded_file.name),'wb') as f:
            # getbuffer():
            # Return a readable and writable view over the contents of the buffer without copying them.
            # Also, mutating the view will transparently update the contents of the buffer:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("Upload Image of a Face")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display image
        display_image = cv2.imread(os.path.join('uploaded_images',uploaded_file.name), cv2.IMREAD_GRAYSCALE)
        st.image(display_image)

        try:
            prediction = helper.predict_emotion(os.path.join(os.getcwd(),'uploaded_images',uploaded_file.name))
            send2trash.send2trash(os.path.join('uploaded_images', uploaded_file.name))

            st.text('Prediction is ' + str(prediction), )

        except:
            st.text('Unable to detect a face in your image, there could be more than one face, please upload again')
    else:
        st.text("Unable to read file")

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port = 8080)