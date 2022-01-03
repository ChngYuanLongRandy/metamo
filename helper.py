import tensorflow as tf
import numpy as np
import os
import cv2

print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f'Current Directory : {os.getcwd()}')

train_root = os.path.join(os.getcwd(), 'train')
test_root = os.path.join(os.getcwd(), 'test')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

hc_path = 'C:\\Users\\Randy\\Venv\\cv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
hc = cv2.CascadeClassifier(hc_path)

model = CNN_v3a_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(48, 48, 1), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(7, activation='softmax')
])

model.load_weights('CNN_v3a_200epochs.h5')

for emotion in emotions:
    f_name = os.listdir(os.path.join(train_root, emotion))
    print(emotion, ' has \t', len(f_name), ' number of files')


def return_cropped_face(img):
    """
    Reads the img, detects a face in the image and returns a cropped image with only the face
    :param img: img with a face
    :return: the cropped version of only the face
    """
    faces = hc.detectMultiScale(img)
    for (x,y,w,h) in faces:
        img = img[y:y+h, x:x+w]
    return img


def predict_emotion(img):
    """
    Predicts the emotion of the img

    :param img: img of a face
    :return: a string representing the emotion of the img
    """

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = return_cropped_face(img)
    img = cv2.resize(img, (48,48))
    prediction = model.predict(img.reshape(1, 48, 48, 1))
    prediction = np.argmax(prediction, axis=-1)

    return emotions[int(prediction)]
