from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import os
from werkzeug.utils import secure_filename

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)
# Create a folder to store temporary user uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- LOAD RNN SEARCH MODEL AND DATA (VERAMEDQuery) ---
# We load these once when the app starts to be efficient
try:
    encoder_model = load_model('medicine_encoder_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    medicine_embeddings = np.load('medicine_embeddings.npy')
    full_df = pd.read_csv('processed_medicines.csv')
    print(">>> VERAMEDQuery model and data loaded successfully.")
except Exception as e:
    print(f">>> ERROR loading VERAMEDQuery model: {e}")
    encoder_model = None

# --- LOAD CNN IMAGE MODEL (VERAMEDScanAI) ---
try:
    cnn_model = load_model('veramed_multiclass_cnn_model.h5')
    # The class names are derived from the folder names in your dataset
    class_names = sorted(os.listdir('VERAMED-AI-DATASET'))
    print(f">>> VERAMEDScanAI model loaded successfully. Found {len(class_names)} classes.")
except Exception as e:
    print(f">>> ERROR loading VERAMEDScanAI model: {e}")
    cnn_model = None
    class_names = []


# --- HELPER FUNCTIONS ---
def clean_text(text):
    """A helper function to clean text for the RNN model."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def find_similar_medicines(query, top_n=10):
    """Processes a search query and returns the top N similar medicines."""
    if not encoder_model or full_df is None:
        return pd.DataFrame() # Return empty if the model or data isn't loaded

    cleaned_query = clean_text(query)
    query_seq = tokenizer.texts_to_sequences([cleaned_query])
    padded_query = pad_sequences(query_seq, maxlen=100, padding='post')
    query_embedding = encoder_model.predict(padded_query, verbose=0) # verbose=0 hides progress bar
    similarities = cosine_similarity(query_embedding, medicine_embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results_df = full_df.iloc[top_indices]
    # Ensure we only try to display columns that actually exist in the dataframe
    display_cols = [col for col in ['Medicine Name', 'Composition', 'Uses'] if col in results_df.columns]
    return results_df[display_cols]


# --- FLASK ROUTES ---
@app.route('/')
def home():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/veramedquery', methods=['GET', 'POST'])
def veramed_query():
    """Handles the medicine search page (VERAMEDQuery)."""
    if request.method == 'POST':
        user_query = request.form['query']
        results_df = find_similar_medicines(user_query)
        # Convert dataframe to a list of dictionaries to easily pass to the template
        results_list = results_df.to_dict(orient='records')
        return render_template('query.html', results=results_list, query=user_query)
    # For a GET request, just show the page with the search bar
    return render_template('query.html', results=None)

@app.route('/veramedscan', methods=['GET', 'POST'])
def veramed_scan():
    """Handles the medicine image scan page (VERAMEDScanAI)."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('scan.html', prediction="Error: No file part in request.")
        file = request.files['file']
        if file.filename == '':
            return render_template('scan.html', prediction="Error: No file selected.")
        if file and cnn_model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image for the CNN model
            img = tf.keras.utils.load_img(filepath, target_size=(180, 180))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            # Make prediction
            predictions = cnn_model.predict(img_array, verbose=0) # verbose=0 hides progress bar
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            
            return render_template('scan.html', prediction=predicted_class)
            
    # For a GET request, just show the page with the upload form
    return render_template('scan.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)

