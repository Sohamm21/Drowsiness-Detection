import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(sound)

classes = ["yawn", "no_yawn", "close", "open", "drinking", "looking_at_passenger", "reaching_behind", "talking_on_phone_left", "talking_on_phone_right"]
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
model = load_model("drowiness_main.h5")
alarm_on = False
alarm_sound = "alarm.mp3"

def sharpen_image(image, factor):
    kernel = np.array([[-1, -1, -1], [-1, 9+factor, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def process_frame(frame, sharpen_factor):
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status1 = ''
    status2 = ''
    
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_color)
        right_eye = right_eye_cascade.detectMultiScale(roi_color)

        for (x1, y1, w1, h1) in left_eye:
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        if status1 == 2 and status2 == 2:
            return "Eyes Closed"
        else:
            sharpened_image = sharpen_image(frame, sharpen_factor)
            return "Eyes Open", sharpened_image

# Streamlit app
def main():
    st.title("Drowsiness Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
    sharpen_factor = st.slider("Adjust Sharpness", 0.0, 5.0, 1.0, 0.1)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        result, sharpened_image = process_frame(image, sharpen_factor)

        st.image(sharpened_image, channels="BGR", caption="Sharpened Image")
        st.write(result)

if __name__ == "__main__":
    main()
