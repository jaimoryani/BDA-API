### Multimodal Depression and PTSD Detection System

This project is a Flask-based web application that predicts Depression and PTSD levels from multimodal inputs, including audio files and transcript files. The system utilizes pre-trained deep learning models for analysis and supports feature extraction from both modalities.

---

## Features

- **Audio Processing**: 
  - Extracts MFCCs, Delta MFCCs, Mel Spectrogram, Chroma features, Tonnetz, and Pitch.
  - Removes silence and standardizes audio length.
  - Segments audio into 50-second clips for better analysis.

- **Text Processing**:
  - Preprocesses transcripts (tokenization, lemmatization, stopword removal).
  - Extracts features using BERT tokenizer and model for embeddings.

- **Model Integration**:
  - Uses pre-trained TensorFlow models (`multimodal_model_depressed.h5` and `multimodal_model_ptsd.h5`) for predictions.
  - Combines audio and text features for final classification.

- **Web Interface**:
  - User-friendly file upload system.
  - Supports selection of models for specific predictions (Depression/PTSD).

---

## Prerequisites

Ensure the following are installed on your system:
- Python 3.8 or higher
- Flask
- TensorFlow
- PyTorch
- Transformers (HuggingFace)
- Librosa
- Pandas
- NLTK
- NumPy

Download additional NLTK resources:
```bash
python -m nltk.downloader punkt stopwords wordnet
```

---

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/jaimoryani/BDA-API
cd BDA-API
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Add Pre-trained Models
- Place `multimodal_model_depressed.h5` and `multimodal_model_ptsd.h5` in the project root directory.

---

## Running the Application

1. Start the Flask server:
   ```bash
   python3 app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload an audio file and a corresponding transcript file. Select the models (Depression/PTSD) for prediction.

---

## File Structure

```
.
├── app.py                  # Main Flask application
├── uploads/                # Directory for storing uploaded files
│   ├── transcripts/        # Sub-directory for transcripts
│   └── audio/              # Sub-directory for audio files
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # HTML template for the web interface
└── README.md               # Project documentation
```

---

## Key Functions

### Audio Feature Extraction
- **`remove_silence(y, sr)`**: Trims silence from audio.
- **`segment_audio(y, sr, segment_length)`**: Splits audio into fixed-length segments.
- **`extract_audio_features(audio_file)`**: Extracts MFCC, delta, chroma, mel spectrogram, tonnetz, and pitch features.

### Text Feature Extraction
- **`preprocess_text(text)`**: Cleans text by lemmatizing and removing stopwords.
- **`extract_text_features(transcript_file)`**: Extracts BERT embeddings from transcripts.

### Prediction
- **`predict_with_model(model, audio_features, text_features)`**: Combines features for model prediction.

---

## Sample API Endpoints

### `GET /`
Renders the homepage.

### `POST /predict`
Handles file uploads and returns prediction results.

#### Input Parameters:
- `transcript`: Transcript CSV file.
- `audio`: Audio file.
- `models`: Selected models (`depressed`, `ptsd`).

#### Output:
```json
{
    "Depression": {
        "label": "Positive",
        "probability": 0.85
    },
    "PTSD": {
        "label": "Negative",
        "probability": 0.25
    }
}
```

---

## Known Issues

- Model inference might be slow for large audio files.
- Ensure the audio and transcript files correspond to the same participant.

---

## Future Enhancements

- Support for real-time predictions using streaming audio.
- Integration of additional mental health prediction models.
- Deployment on cloud platforms for scalability.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- TensorFlow and PyTorch communities.
- HuggingFace for the BERT model.
- Librosa for audio processing utilities.

```
