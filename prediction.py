# -*- coding: utf-8 -*-
"""
Created on sat march  1 17:20:13 2020

@author: megha
"""
import cv2
import tensorflow as tf
CATEGORIES = ["Normal"]
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")
image = prepare("CHNCXR_0003_0.png") #your image path
prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])