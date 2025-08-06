from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sys
from werkzeug.utils import secure_filename

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your processing function
from censor_audi_simple import censor_audio_file

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'app', 'processed')
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if not allowed_file(file.filename):
            return "Invalid file type. Only MP3 and WAV are allowed.", 400
        
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        output_filename = f"censored_{filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        try:
            censor_audio_file(upload_path, output_path)
        except Exception as e:
            return f"Error processing audio: {str(e)}", 500

        return redirect(url_for('download_ready', filename=output_filename))

    return render_template('upload.html')

@app.route('/download_ready/<filename>')
def download_ready(filename):
    return render_template('download_ready.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# Main
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
