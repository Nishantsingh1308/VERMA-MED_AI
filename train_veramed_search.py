import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# --- 1. DEFINE STANDARDIZED COLUMN NAMES ---
MEDICINE_NAME_COL = 'Medicine Name'
COMPOSITION_COL = 'Composition'
USES_COL = 'Uses'

def train_and_save_model():
    """
    This function contains the entire training pipeline. It loads data,
    builds the model, and saves the necessary files.
    """
    # --- 2. LOAD AND PREPROCESS DATA ---
    print("--- Starting Model Training ---")
    print("Loading and standardizing data...")
    df_details = pd.read_csv('Medicine_details.csv', encoding='latin1')
    df_details = df_details.rename(columns={
        'Medicine Name': MEDICINE_NAME_COL, 'Composition': COMPOSITION_COL, 'Manufacturer': USES_COL
    })
    df_details = df_details[[MEDICINE_NAME_COL, COMPOSITION_COL, USES_COL]]

    df_indian = pd.read_csv('Indian_medicine_data.csv', encoding='latin1')
    df_indian = df_indian.rename(columns={
        'name': MEDICINE_NAME_COL, 'salt_composition': COMPOSITION_COL, 'medicine_desc': USES_COL
    })
    df_indian = df_indian[[MEDICINE_NAME_COL, COMPOSITION_COL, USES_COL]]

    full_df = pd.concat([df_details, df_indian], ignore_index=True)
    full_df.fillna('', inplace=True)
    full_df['search_text'] = full_df[MEDICINE_NAME_COL] + ' ' + full_df[COMPOSITION_COL] + ' ' + full_df[USES_COL]
    print(f"Successfully combined data into {len(full_df)} records.")

    # --- 3. TOKENIZE TEXT ---
    print("Tokenizing text...")
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    full_df['cleaned_text'] = full_df['search_text'].apply(clean_text)

    MAX_VOCAB_SIZE = 10000
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<unk>')
    tokenizer.fit_on_texts(full_df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(full_df['cleaned_text'])

    MAX_SEQ_LENGTH = 100
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

    # --- 4. BUILD THE RNN (LSTM) FOR EMBEDDINGS ---
    print("Building model...")
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    EMBEDDING_DIM = 64
    LATENT_DIM = 32

    input_seq = Input(shape=(MAX_SEQ_LENGTH,))
    embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(input_seq)
    lstm_layer = LSTM(LATENT_DIM)(embedding_layer)
    encoder_model = Model(input_seq, lstm_layer)
    encoder_model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model built successfully.")

    # --- 5. GENERATE AND SAVE EMBEDDINGS ---
    print("Generating and saving embeddings...")
    medicine_embeddings = encoder_model.predict(padded_sequences)

    output_files = ['medicine_embeddings.npy', 'processed_medicines.csv', 'tokenizer.pickle', 'medicine_encoder_model.h5']
    for f in output_files:
        if os.path.exists(f): os.remove(f)

    np.save('medicine_embeddings.npy', medicine_embeddings)
    full_df.to_csv('processed_medicines.csv', index=False)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    encoder_model.save('medicine_encoder_model.h5')

    print("\n--- Model Training Complete! ---")
    for f in output_files: print(f"Saved new file: {f}")
    
    # Return all necessary components for the interactive session
    return encoder_model, tokenizer, medicine_embeddings, full_df, clean_text, MAX_SEQ_LENGTH


def interactive_query_session(encoder_model, tokenizer, medicine_embeddings, full_df, clean_text_func, max_len):
    """
    Starts an interactive session in the terminal to test the search model.
    """
    print("\n--- Starting Interactive Query Session ---")
    print("Type your search query and press Enter. Type 'exit' or 'quit' to stop.")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting query session. Goodbye!")
            break
        
        # Process the query using the loaded models
        cleaned_query = clean_text_func(query)
        query_seq = tokenizer.texts_to_sequences([cleaned_query])
        padded_query = pad_sequences(query_seq, maxlen=max_len, padding='post')
        
        query_embedding = encoder_model.predict(padded_query, verbose=0)
        
        similarities = cosine_similarity(query_embedding, medicine_embeddings).flatten()
        
        top_indices = similarities.argsort()[-10:][::-1] # Top 10 results
        
        results_df = full_df.iloc[top_indices]
        
        print("\n--- Top 10 Search Results ---")
        if results_df.empty:
            print("No similar medicines found.")
        else:
            # Print in a clean, readable format
            print(results_df[[MEDICINE_NAME_COL, COMPOSITION_COL, USES_COL]].to_string(index=False))


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Step 1: Run the training and save all the files for the website
    components = train_and_save_model()
    
    # Step 2: Use the returned components to start the interactive session
    interactive_query_session(*components)

