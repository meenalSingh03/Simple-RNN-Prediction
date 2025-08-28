# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import h5py
import json
import streamlit as st

# Custom function to load model with compatibility fixes
def load_model_with_compatibility(filepath):
    try:
        return tf.keras.models.load_model(filepath)
    except ValueError as e:
        if 'time_major' in str(e) or 'Unrecognized keyword arguments' in str(e):
            st.write("Fixing compatibility issue with time_major parameter...")
            
            with h5py.File(filepath, 'r') as f:
                model_config_raw = f.attrs['model_config']
                # Handle both string and bytes
                if isinstance(model_config_raw, bytes):
                    model_config_str = model_config_raw.decode('utf-8')
                else:
                    model_config_str = model_config_raw
                model_config = json.loads(model_config_str)
                
                def clean_config(config):
                    if isinstance(config, dict):
                        if 'config' in config and isinstance(config['config'], dict):
                            config['config'].pop('time_major', None)
                        for key, value in config.items():
                            config[key] = clean_config(value)
                    elif isinstance(config, list):
                        for i, item in enumerate(config):
                            config[i] = clean_config(item)
                    return config
                
                cleaned_config = clean_config(model_config)
                model = tf.keras.models.model_from_json(json.dumps(cleaned_config))
                model.load_weights(filepath)
                st.write("Model loaded successfully with compatibility fixes!")
                return model
        else:
            raise e

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation (using compatibility function)
model = load_model_with_compatibility('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.4f}')
    else:
        st.write('Please enter a movie review.')
else:
    st.write('Please enter a movie review.')
