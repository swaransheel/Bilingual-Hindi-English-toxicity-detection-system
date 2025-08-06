# Bilingual Hindi-English Toxicity Detection System

A sophisticated AI-powered web application that detects and censors toxic content in audio files. The system supports both Hindi and English languages, providing real-time audio processing with an intuitive web interface.

## 🌟 Features

- **Bilingual Support**: Detects toxic content in both Hindi and English
- **Real-time Processing**: Advanced audio processing with word-level timestamps
- **Web Interface**: User-friendly Flask-based web application
- **AI-Powered**: Uses state-of-the-art models including Whisper for transcription and transformer models for toxicity detection
- **Audio Formats**: Supports MP3 and WAV audio files
- **Smart Censoring**: Replaces toxic words with beep sounds while preserving audio timing
- **Download Ready**: Processed files can be downloaded instantly

## 🚀 Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: 
  - Faster-Whisper for speech-to-text transcription
  - Transformers for toxicity classification
  - PyTorch for deep learning operations
- **Audio Processing**: PyDub for audio manipulation
- **Frontend**: HTML, Tailwind CSS
- **Language Detection**: Automatic language identification
- **Natural Language Processing**: NLTK for text processing

## 🤖 AI Models Used

The system employs multiple state-of-the-art AI models for comprehensive toxicity detection:

### Speech Recognition
- **Faster-Whisper**: High-performance implementation of OpenAI's Whisper model
  - Supports multiple model sizes: `tiny`, `base`, `small`, `medium`, `large`
  - Provides word-level timestamps for precise audio censoring
  - Automatic language detection for Hindi and English

### Toxicity Detection Models

#### English Language Models
- **unitary/toxic-bert**: Primary toxicity detection model
  - BERT-based model fine-tuned for toxic content detection
  - High accuracy for English toxic language patterns

- **facebook/roberta-hate-speech-dynabench-r4-target**: Hate speech detection
  - RoBERTa model specialized in identifying hate speech
  - Trained on diverse hate speech datasets

- **DunnBC22/ibert-roberta-base-Abusive_Or_Threatening_Speech**: Abusive language detection
  - iBERT model optimized for detecting abusive and threatening content
  - Focuses on aggressive language patterns

- **distilbert-base-uncased-finetuned-sst-2-english**: Sentiment analysis
  - DistilBERT model for sentiment classification
  - Helps in context-aware toxicity detection

- **Newtral/xlm-r-finetuned-toxic-political-tweets-es**: Political toxicity
  - Multilingual RoBERTa model for political content moderation
  - Specialized in detecting toxic political discourse

#### Hindi/Hinglish Language Models
- **Hate-speech-CNERG/hindi-codemixed-abusive-MuRIL**: Hindi abusive content detection
  - MuRIL (Multilingual Representations for Indian Languages) based model
  - Specifically designed for Hindi and code-mixed (Hinglish) content
  - Trained on Indian social media data

- **pascalrai/hinglish-twitter-roberta-base-sentiment**: Hinglish sentiment analysis
  - RoBERTa model fine-tuned for Hinglish sentiment analysis
  - Understands code-switching between Hindi and English

### Language Detection
- **langid.py**: Automatic language identification
  - Determines whether content is primarily Hindi or English
  - Enables appropriate model selection for toxicity detection

### Model Architecture Benefits
- **Ensemble Approach**: Multiple models work together for higher accuracy
- **Language-Specific**: Dedicated models for Hindi and English content
- **Context-Aware**: Considers sentiment and political context
- **Multilingual Support**: Handles code-mixed (Hinglish) content
- **Real-time Processing**: Optimized for fast inference

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- **Minimum 4GB RAM** (8GB+ recommended for optimal performance)
- **GPU support** (optional but recommended):
  - NVIDIA GPU with CUDA support for faster model inference
  - Alternatively, the system will use CPU (slower but functional)
- **Internet connection** for initial model downloads (~2-5GB total)

### Model Download Requirements
On first run, the system will automatically download the following models:
- Faster-Whisper models (100MB - 3GB depending on size)
- Transformers models (~1-2GB total for all toxicity detection models)
- NLTK data packages (~50MB)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/swaransheel/Bilingual-Hindi-English-toxicity-detection-system.git
   cd Bilingual-Hindi-English-toxicity-detection-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install manually:
   ```bash
   pip install flask pydub faster-whisper transformers torch langid-py nltk werkzeug
   ```

4. **Download NLTK data** (if needed)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 🎯 Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

3. **Upload and Process Audio**
   - Click on "Upload Audio File" 
   - Select an MP3 or WAV file
   - Wait for processing to complete
   - Download the censored audio file

### Command Line Usage

You can also use the core functionality directly:

```python
from censor_audi_simple import censor_audio_file

# Process an audio file
input_path = "path/to/your/audio.mp3"
output_path = "path/to/censored/audio.mp3"
censor_audio_file(input_path, output_path)
```

## 📁 Project Structure

```
Bilingual-Hindi-English-toxicity-detection-system/
├── app.py                          # Flask web application
├── censor_audi_simple.py          # Core audio censoring logic
├── __Init__.py                     # Package initialization
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── landing.html               # Homepage
│   ├── upload.html                # File upload page
│   └── download_ready.html        # Download page
├── uploads/                        # Uploaded audio files
├── processed/                      # Processed (censored) audio files
└── README.md                      # This file
```

## 🔧 Configuration

The application uses the following default configurations:

- **Upload folder**: `uploads/`
- **Processed folder**: `processed/`
- **Allowed file types**: MP3, WAV
- **Host**: `0.0.0.0` (accessible from all network interfaces)
- **Debug mode**: Enabled (disable in production)

### Model Configuration

#### Whisper Model Sizes
- **tiny**: Fastest, least accurate (~39MB)
- **base**: Good balance of speed and accuracy (~74MB) - Default
- **small**: Better accuracy (~244MB)
- **medium**: High accuracy (~769MB)
- **large**: Highest accuracy (~1550MB)

#### Toxicity Detection Thresholds
- **Default threshold**: 0.4 (adjustable)
- **Minimum votes**: 2 models must agree for toxicity classification
- **Ensemble voting**: Multiple models vote on toxicity classification

#### Device Configuration
- **CPU**: Default fallback, slower processing
- **GPU**: Automatic detection if CUDA available, significantly faster
- **Memory management**: Automatic cleanup after processing

## 🧠 How It Works

1. **Audio Upload**: Users upload audio files through the web interface
2. **Transcription**: Faster-Whisper converts speech to text with word-level timestamps
3. **Language Detection**: Automatic identification of Hindi/English content
4. **Toxicity Detection**: AI models analyze text for toxic content
5. **Audio Censoring**: Toxic words are replaced with beep sounds
6. **File Generation**: Censored audio file is generated and made available for download

## 🎨 Web Interface

The web application features:

- **Landing Page**: Welcome screen with application overview
- **Upload Interface**: Drag-and-drop file upload with format validation
- **Processing Status**: Real-time feedback during audio processing
- **Download Page**: Secure download of processed files

## 🔒 Security Features

- **File Validation**: Only allows MP3 and WAV files
- **Secure Filenames**: Uses werkzeug's secure_filename for file handling
- **Error Handling**: Comprehensive error handling for robustness
- **Path Security**: Prevents directory traversal attacks

## 🚨 Error Handling

The application handles various error scenarios:

- Invalid file formats
- Missing files
- Processing errors
- Network issues
- Memory limitations

## 🔧 Customization

### Adding New Languages

To add support for additional languages:

1. Update the language detection logic in `censor_audi_simple.py`
2. Add language-specific toxic word lists
3. Configure appropriate tokenization and preprocessing

### Modifying Toxicity Thresholds

Adjust toxicity detection sensitivity by modifying the threshold values in the toxicity classification pipeline.

### Changing Audio Processing

Customize beep generation, fade effects, and audio quality settings in the audio processing functions.

## 📊 Performance Considerations

- **GPU Support**: Automatically uses GPU if available for faster processing
- **Memory Management**: Efficient memory cleanup after processing
- **File Size**: Larger audio files will take longer to process
- **Concurrent Users**: Consider using production WSGI server for multiple users

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Swaransheel**
- GitHub: [@swaransheel](https://github.com/swaransheel)

## 🙏 Acknowledgments

- **OpenAI Whisper team** for the excellent speech recognition model
- **Hugging Face** for providing transformer models and the transformers library
- **PyDub team** for audio processing capabilities
- **Flask community** for the web framework

### Model Contributors
- **Unitary AI** - toxic-bert model for toxicity detection
- **Facebook Research** - RoBERTa hate speech detection model
- **DunnBC22** - iBERT abusive speech detection model
- **Newtral** - XLM-R political toxicity model
- **CNERG (IIT Kharagpur)** - Hindi code-mixed abusive content detection
- **Pascal Rai** - Hinglish sentiment analysis model
- **Google Research** - MuRIL (Multilingual Representations for Indian Languages)
- **Distilbert team** - Sentiment analysis model

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/swaransheel/Bilingual-Hindi-English-toxicity-detection-system/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer through GitHub

## 🔄 Updates

This project is actively maintained. Check the [Releases](https://github.com/swaransheel/Bilingual-Hindi-English-toxicity-detection-system/releases) page for the latest updates and features.

---

**⚠️ Disclaimer**: This tool is designed for content moderation purposes. The accuracy of toxicity detection may vary based on context, slang, and evolving language patterns. Always review results for critical applications.
