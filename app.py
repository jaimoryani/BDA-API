import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertModel
import librosa
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)

# Load the pre-trained models
depressed_model = tf.keras.models.load_model('multimodal_model_depressed.h5')
ptsd_model = tf.keras.models.load_model('multimodal_model_ptsd.h5')

# Define dimensions from model training
AUDIO_DIM = 193
TEXT_DIM = 768

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stop words and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(cleaned_tokens)

# Function to remove silence from the audio
def remove_silence(y, sr):
    '''Remove silence from the beginning and end of an audio clip.'''
    yt, _ = librosa.effects.trim(y)
    return yt

# Function to standardize audio length by padding or trimming
def standardize_audio_length(y, sr, target_length=5):
    '''Standardize audio length to target length in seconds.'''
    target_samples = target_length * sr
    if len(y) < target_samples:
        # Pad with zeros
        y = np.pad(y, (0, target_samples - len(y)), 'constant')
    else:
        # Trim to the target length
        y = y[:target_samples]
    return y

# Function to segment audio into fixed length (50 seconds)
def segment_audio(y, sr, segment_length=50):
    '''Segment audio into fixed length (50 seconds).'''
    segment_samples = int(segment_length * sr)
    return [y[i:i + segment_samples] for i in range(0, len(y), segment_samples)]

# Function to extract summarized audio features from each segment
def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Remove silence from the beginning and end
    y = remove_silence(y, sr)
   
    # Segment audio into 50s clips
    segments = segment_audio(y, sr)
    
    all_features = []
    
    # Extract features for each segment
    for segment in segments:
        # Initialize result array for stacking features
        res = np.array([])

        # MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T, axis=0)
        res = np.hstack((res, mfccs))
        print("MFCCs shape:", mfccs.shape)

        # Delta MFCCs
        delta_mfcc = librosa.feature.delta(mfccs)
        res = np.hstack((res, delta_mfcc))
        print("Delta MFCC shape:", delta_mfcc.shape)

        # Double Delta MFCCs
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        res = np.hstack((res, delta2_mfcc))
        print("Delta2 MFCC shape:", delta2_mfcc.shape)
        
        # Chroma features
        stft = np.abs(librosa.stft(segment))  # Ensure STFT is calculated
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        res = np.hstack((res, chroma))
        print("Chroma shape:", chroma.shape)
        
        # Mel Spectrogram
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr).T, axis=0)
        res = np.hstack((res, mel_spectrogram))
        print("Mel Spectrogram shape:", mel_spectrogram.shape)
        
        # Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        res = np.hstack((res, contrast))
        print("Contrast shape:", contrast.shape)
        
        # Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=segment, sr=sr).T, axis=0)
        res = np.hstack((res, tonnetz))
        print("Tonnetz shape:", tonnetz.shape)
        
        # Zero Crossing Rate (mean value)
        #zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=segment).T, axis=0)
        #res = np.hstack((res, [zero_crossing_rate]))
        #print("Zero Crossing Rate shape:", zero_crossing_rate.shape)
        
        # Pitch extraction
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
        # Use the maximum pitch value from the track
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        res = np.hstack((res, [pitch]))
        print("Pitch shape:", np.array([pitch]).shape)  # Print shape as a single element

        all_features.append(res)

    return all_features


# Function to extract text features using BERT tokenizer and embeddings
def extract_text_features(transcript_file):
    # Load transcript file
    transcript_df = pd.read_csv(transcript_file)
    
    # Concatenate all participant text (assuming all text is from participants)
    participant_lines = transcript_df['Text']
    all_text = " ".join(participant_lines)
    
    # Preprocess the text
    cleaned_text = preprocess_text(all_text)
    
    # Tokenize and use BERT embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Tokenize the text
    inputs = tokenizer(cleaned_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
        print("BERT embedding shape:", embedding.shape)
    
    ## Prepare feature dictionary with BERT embeddings
    #features = {f'text_feature_{i+1}': val for i, val in enumerate(embedding)}
    
    return embedding
    
def extract_features(audio_path, transcript_path):
    """Extract multimodal features from audio and transcript."""
    # Extract audio features
    audio_features = extract_audio_features(audio_path)
    text_features = extract_text_features(transcript_path)

    return audio_features, text_features

def predict_with_model(model, audio_features, text_features):
    all_predictions = np.array([])
    
    for row in audio_features:
    	"""Make a prediction using a loaded model."""
    	audio_input = np.array(row).reshape(1, -1)  # Ensure it's (1, 193)
    	text_input = np.array(text_features).reshape(1, -1)  # Ensure it's (1, 768)
    	# Perform prediction with the model
    	prediction = model.predict([text_input, audio_input])[0][0]
    	all_predictions = np.hstack((all_predictions, prediction))
    
    print(all_predictions)
    final_pred = np.mean(all_predictions)
    
    return "Positive" if final_pred > 0.5 else "Negative", float(final_pred)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file uploads, extract features, and make predictions."""
    if 'transcript' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Please upload both transcript and audio files"}), 400

    transcript_file = request.files['transcript']
    audio_file = request.files['audio']

    # Ensure directories exist
    transcript_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'transcripts')
    audio_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'audio')

    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Save uploaded files
    transcript_path = os.path.join(transcript_dir, secure_filename(transcript_file.filename))
    audio_path = os.path.join(audio_dir, secure_filename(audio_file.filename))

    try:
        transcript_file.save(transcript_path)
        audio_file.save(audio_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

    # Extract features
    try:
        audio_features, text_features = extract_features(audio_path, transcript_path)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    selected_models = request.form.getlist('models')
    results = {}

    # Perform predictions for both models if selected
    if 'depressed' in selected_models:
        label, prob = predict_with_model(depressed_model, audio_features, text_features)
        results['Depression'] = {"label": label, "probability": prob}

    if 'ptsd' in selected_models:
        label, prob = predict_with_model(ptsd_model, audio_features, text_features)
        results['PTSD'] = {"label": label, "probability": prob}

    # Clean up uploaded files
    os.remove(transcript_path)
    os.remove(audio_path)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

