from flask import Flask, request, jsonify, render_template, url_for
import os
import re
import tempfile
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from extractor import extract_text_from_video, extract_text_from_url
from summarizer import summarize_text, summarize_youtube_video, summarize_video_direct, MAX_VIDEO_SIZE, SUPPORTED_VIDEO_FORMATS

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')

def extract_video_id(url):
    """Extract YouTube video ID from various YouTube URL formats."""
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be|in)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    match = re.search(youtube_regex, url)
    return match.group(6) if match else None

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure file size limits
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Configure text length limits
MAX_TEXT_LENGTH = 10000  # Example maximum text length
MAX_VIDEO_URL_LENGTH = 200  # Example maximum URL length

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    # Check if file part is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files["file"]
    
    # Check if the file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload a video file (MP4, AVI, MOV, etc)."}), 400
    
    try:
        # Securely save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Try direct video processing first
        if os.path.getsize(filepath) <= MAX_VIDEO_SIZE:
            ext = os.path.splitext(filename)[1].lower()[1:]
            if ext in SUPPORTED_VIDEO_FORMATS:
                summary = summarize_video_direct(filepath)
                if summary:
                    os.remove(filepath)
                    return jsonify({"summary": summary})

        # Fall back to speech recognition if direct processing fails
        text = extract_text_from_video(filepath)
        
        # Check text length
        if len(text) > MAX_TEXT_LENGTH:
            os.remove(filepath)
            return jsonify({"error": f"Extracted text exceeds the maximum allowed length of {MAX_TEXT_LENGTH} characters."}), 400

        # Summarize text
        summary = summarize_text(text)
        
        # Clean up the file
        os.remove(filepath)
        
        return jsonify({"summary": summary})
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/summarize-url", methods=["POST"])
def summarize_url():
    try:
        data = request.get_json()
        
        if not data or "url" not in data:
            return jsonify({"error": "No URL provided."}), 400
        
        url = data["url"].strip()
        
        if not url:
            return jsonify({"error": "URL cannot be empty."}), 400
        
        # Check URL length
        if len(url) > MAX_VIDEO_URL_LENGTH:
            return jsonify({"error": f"URL exceeds the maximum allowed length of {MAX_VIDEO_URL_LENGTH} characters."}), 400
        
        # Check if URL is valid
        try:
            response = requests.head(url, timeout=5)
            if response.status_code >= 400:
                return jsonify({"error": f"URL returned status code {response.status_code}."}), 400
        except requests.exceptions.RequestException:
            return jsonify({"error": "Invalid or inaccessible URL."}), 400

        # Check if the URL is a YouTube link
        if "youtube.com" in url or "youtu.be" in url:
            video_id = extract_video_id(url)
            if video_id:
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                summary = summarize_youtube_video(youtube_url)
            else:
                return jsonify({"error": "Invalid YouTube URL format."}), 400
        elif url.endswith(tuple(f".{ext}" for ext in ALLOWED_EXTENSIONS)):
            # Handle video file URLs
            try:
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(url)[1], delete=False) as temp_file:
                    video_response = requests.get(url, stream=True, timeout=30)
                    video_response.raise_for_status()
                    
                    total_size = 0
                    for chunk in video_response.iter_content(chunk_size=8192):
                        total_size += len(chunk)
                        if total_size > MAX_CONTENT_LENGTH:
                            os.unlink(temp_file.name)
                            return jsonify({"error": f"Video file exceeds the maximum allowed size of {MAX_CONTENT_LENGTH / (1024 * 1024)} MB."}), 400
                        temp_file.write(chunk)
                
                # Extract text from downloaded video
                text = extract_text_from_video(temp_file.name)
                
                # Check text length
                if len(text) > MAX_TEXT_LENGTH:
                    os.unlink(temp_file.name)
                    return jsonify({"error": f"Extracted text exceeds the maximum allowed length of {MAX_TEXT_LENGTH} characters."}), 400
                
                # Summarize text
                summary = summarize_text(text)
                
                # Clean up
                os.unlink(temp_file.name)
            except Exception as e:
                return jsonify({"error": f"Error processing video URL: {str(e)}"}), 500
        else:
            # Process as a regular webpage
            text = extract_text_from_url(url)
            
            # Check text length
            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({"error": f"Extracted text exceeds the maximum allowed length of {MAX_TEXT_LENGTH} characters."}), 400
            
            # Summarize text
            summary = summarize_text(text)
        
        return jsonify({"summary": summary})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/summarize-youtube", methods=["POST"])
def summarize_youtube():
    try:
        data = request.get_json()
        
        if not data or "url" not in data:
            return jsonify({"error": "No YouTube URL provided."}), 400
        
        youtube_url = data["url"].strip()
        
        if not youtube_url:
            return jsonify({"error": "YouTube URL cannot be empty."}), 400
        
        # Summarize the YouTube video
        summary = summarize_youtube_video(youtube_url)
        return jsonify({"summary": summary})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "The file is too large. Maximum file size is 50MB."}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error. Please try again later."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)