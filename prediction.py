import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import rembg, PIL, io, joblib, keras
from keras.applications import resnet50


# Inputs
saved_model = "resnet50_model.h5"
preprocess_method = resnet50.preprocess_input
saved_class_index_to_label = "class_index_to_label.sav"


# Prevent loading model multiple times
# by caching function using @st.cache_resource
@st.cache_resource(show_spinner= False)
def load_model():
    '''
    Load previously saved model
    extension .h5 is faster than .keras
    Also load saved class index-to-label converter
    '''
    model = keras.models.load_model(saved_model)
    class_index_to_label = joblib.load(saved_class_index_to_label)
    return model, class_index_to_label


@st.cache_resource(show_spinner= False)
def load_lightweight_model():
    '''
    FOR FASTER MODEL LOADING
    Load lightweight version of saved model 
    that was previously saved specifically using model.export()
    The only available attribute is model.serve()
    to make prediction equivalent of model.predict()
    https://www.tensorflow.org/guide/keras/serialization_and_saving#exporting
    '''
    # Load lightweight saved model-limited functionality 
    model = tf.saved_model.load(saved_model)

    class_index_to_label = joblib.load(saved_class_index_to_label)
    return model, class_index_to_label


def remove_background(image_path, display= False):
    '''
    Replace background with plain white
    using PIL and rembg libraries
    Return io.BytesIO temp path of converted img
    Optionally display converted img
    '''

    # Load the input image
    input = PIL.Image.open(image_path)

    # Replace background with white 
    output = rembg.remove(input, bgcolor=(255,255,255,255))

    # Convert from RGBA to RGB
    output_rgb = output.convert('RGB')

    # Display converted image
    if display:
        plt.imshow(output_rgb)
        plt.grid(False)

    # Create temp path in memory to save bg-removed img
    temp_path = io.BytesIO()

    # Save to temp path
    output_rgb.save(temp_path, format= 'jpeg')

    return temp_path


def predict(image_path):
    '''
    Loads model using load_model()
    Makes class label prediction and % confidence
    Returns [predicted class label, confidence]
    '''
    model, class_index_to_label = load_model()
    target_size = model.input_shape[1:3]

    img = keras.utils.load_img(image_path,
                            target_size= target_size,
                            keep_aspect_ratio= True, # crop in the center
                            )

    # convert img object to 3D np.array
    # -> (height,width,channel)
    img_array = keras.utils.img_to_array(img)

    # preprocess img_array using the provided preprocess_method
    preprocessed_array = preprocess_method(img_array)

    # model expects 4-dim tensor input, so add an extra dim
    # -> (1,height,width,channel)
    img_tensor = np.expand_dims(preprocessed_array, axis=0)

    # predict class probs
    pred_probs = model.predict(img_tensor)

    # get the most likely class index
    pred_class_index = np.argmax(pred_probs, axis=-1)[0]

    # class index --> label
    pred_class_label = class_index_to_label[pred_class_index]

    # prediction confidence 
    confidence = int(pred_probs[0, pred_class_index] *100)

    return [pred_class_label, confidence]