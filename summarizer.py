import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

# Explicitly load the key.env file from the video_summarizer folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'key.env'))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

# Define API limitations
MAX_TEXT_LENGTH = 5000  # Example limit for text input
MAX_VIDEO_URL_LENGTH = 200  # Example limit for video URL length
MAX_VIDEO_SIZE = 20 * 1024 * 1024  # 20MB Gemini limit
SUPPORTED_VIDEO_FORMATS = ['mp4', 'mov', 'webm']

def get_video_base64(video_path):
    """Convert video to base64 string."""
    with open(video_path, 'rb') as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def summarize_video_direct(video_path):
    """Directly summarize video using Gemini if within size limits."""
    try:
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > MAX_VIDEO_SIZE:
            return None  # Too large for direct processing
            
        # Check format
        ext = os.path.splitext(video_path)[1].lower()[1:]
        if ext not in SUPPORTED_VIDEO_FORMATS:
            return None  # Unsupported format
            
        # Convert video to base64
        video_base64 = get_video_base64(video_path)
        
        # Create video message
        prompt = """
        Please analyze this video and provide a detailed summary covering:
        - Main content and subject matter
        - Key events or information presented
        - Important details or highlights
        - Overall context and significance
        Make the summary clear and well-structured.
        """
        
        # Generate summary using Gemini
        response = model.generate_content([
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": f"video/{ext}",
                            "data": video_base64
                        }
                    }
                ]
            }
        ])
        
        return response.text if response else "Failed to generate summary."
    except Exception as e:
        return f"Error processing video directly: {str(e)}"

def summarize_text(text):
    """
    Summarize provided text using Gemini AI.
    
    Args:
        text (str): The text to summarize
        
    Returns:
        str: The summarized text
    """
    if not text or not text.strip():
        return "Please provide valid text to summarize."
    
    if len(text) > MAX_TEXT_LENGTH:
        # Consider chunking large text for better handling
        chunks = [text[i:i+MAX_TEXT_LENGTH] for i in range(0, len(text), MAX_TEXT_LENGTH)]
        summaries = []
        
        for chunk in chunks:
            try:
                prompt = f"""
                Please provide a concise summary of the following text. 
                Focus on the key points and main ideas:
                
                {chunk}
                """
                response = model.generate_content(prompt)
                summaries.append(response.text)
            except Exception as e:
                summaries.append(f"Error summarizing text chunk: {str(e)}")
        
        # Combine chunk summaries
        if len(summaries) > 1:
            combined_summary = "\n\n".join(summaries)
            return f"Summary (from {len(chunks)} text segments):\n\n{combined_summary}"
        return summaries[0]
    
    try:
        prompt = f"""
        Please provide a concise summary of the following text.
        Focus on the key points and main ideas:
        
        {text}
        """
        response = model.generate_content(prompt)
        return response.text if response else "Failed to generate summary."
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def get_youtube_info(video_url):
    """Get YouTube video information and transcript."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
            
            title = video_info.get('title', '')
            description = video_info.get('description', '')
            duration = video_info.get('duration', 0)
            
            # Get transcript
            video_id = video_info.get('id', '')
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript_text = ' '.join([entry['text'] for entry in transcript])
            except:
                transcript_text = ''
                
            return {
                'title': title,
                'description': description,
                'duration': duration,
                'transcript': transcript_text
            }
    except Exception as e:
        print(f"Error getting YouTube info: {str(e)}")
        return None

def summarize_youtube_video(youtube_url):
    """Summarizes a YouTube video using its content and transcript."""
    if len(youtube_url) > MAX_VIDEO_URL_LENGTH:
        return f"URL exceeds the maximum allowed length of {MAX_VIDEO_URL_LENGTH} characters."
    
    try:
        # Get video information
        video_info = get_youtube_info(youtube_url)
        if not video_info:
            return "Failed to fetch video information. Please check the URL and try again."

        # Construct detailed prompt with video information
        prompt = f"""
        Provide a comprehensive summary of this specific YouTube video:
        
        Title: {video_info['title']}
        Duration: {video_info['duration']} seconds
        
        Video Description:
        {video_info['description']}
        
        Video Transcript:
        {video_info['transcript'][:5000]}
        
        Please analyze this content and provide:
        1. Main topic and key points discussed in the video
        2. Important details, examples, and demonstrations shown
        3. Key arguments or explanations presented
        4. Conclusions or takeaways from the video
        
        Focus only on the actual content of this specific video.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip() if response else "Failed to generate summary."
        
    except Exception as e:
        return f"Error summarizing YouTube video: {str(e)}"