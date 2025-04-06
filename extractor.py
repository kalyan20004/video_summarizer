import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import tempfile

def extract_text_from_url(url):
    """Extracts text from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join([p.get_text() for p in paragraphs])

def process_audio_chunk(recognizer, chunk):
    """Process individual audio chunk for better recognition."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_chunk:
            chunk.export(temp_chunk.name, format='wav', 
                        parameters=["-ac", "1", "-ar", "16000"])
            with sr.AudioFile(temp_chunk.name) as source:
                audio = recognizer.record(source)
                return recognizer.recognize_google(audio, language='en-US')
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise Exception(f"API unavailable: {str(e)}")

def extract_text_from_video(video_path):
    """Extracts speech text from a video file."""
    try:
        # Check file exists
        if not os.path.exists(video_path):
            return "Error: Video file not found"

        # Get file extension
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']:
            return "Error: Unsupported video format. Please use MP4, AVI, MOV, WMV, FLV, or MKV"

        # Convert video to audio
        audio = AudioSegment.from_file(video_path)
        
        # Convert stereo to mono
        audio = audio.set_channels(1)
        
        # Set sample rate to 16kHz
        audio = audio.set_frame_rate(16000)
        
        # Normalize audio volume
        audio = audio.normalize()
        
        # Split audio on silence for better processing
        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS - 14,
            keep_silence=500
        )
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Process each chunk
        text_chunks = []
        for chunk in chunks:
            try:
                chunk_text = process_audio_chunk(recognizer, chunk)
                if chunk_text:
                    text_chunks.append(chunk_text)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        if not text_chunks:
            return "No speech could be recognized in the video"
            
        return " ".join(text_chunks)

    except Exception as e:
        return f"Error processing video: {str(e)}"
