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
            
            # Create a temporary copy of the model with fixed configuration
            import tempfile
            import shutil
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Copy the original file
            shutil.copy2(filepath, temp_path)
            
            try:
                # Modify the model config in the temporary file
                with h5py.File(temp_path, 'r+') as f:
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
                                if isinstance(value, (dict, list)):
                                    clean_config(value)
                        elif isinstance(config, list):
                            for item in config:
                                if isinstance(item, (dict, list)):
                                    clean_config(item)
                    
                    clean_config(model_config)
                    
                    # Update the model config in the file
                    cleaned_config_str = json.dumps(model_config)
                    if isinstance(model_config_raw, bytes):
                        f.attrs['model_config'] = cleaned_config_str.encode('utf-8')
                    else:
                        f.attrs['model_config'] = cleaned_config_str
                
                # Load the model from the modified temporary file
                model = tf.keras.models.load_model(temp_path)
                st.write("Model loaded successfully with compatibility fixes!")
                return model
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass
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
    # Limit vocabulary to match training (typically 10000 most common words)
    # Index 0: padding, 1: start, 2: unknown, 3: unused
    # So word indices start from 4 and go up to 10003 (10000 + 4 - 1)
    encoded_review = []
    for word in words:
        word_idx = word_index.get(word, 2)  # 2 = unknown word
        # Ensure the index doesn't exceed vocabulary size (10000 + offset)
        if word_idx < 10000:  # Keep only indices within vocab range
            encoded_review.append(word_idx + 3)
        else:
            encoded_review.append(2 + 3)  # Use unknown token for out-of-vocab words
    
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
