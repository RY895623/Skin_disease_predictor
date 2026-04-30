import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image: Image.Image):
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_condition(model, image: Image.Image, class_names):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], confidence
