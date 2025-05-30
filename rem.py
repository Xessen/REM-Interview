import streamlit as st
import requests
from pydub import AudioSegment
import os
import tempfile
import torchaudio
import whisper
from textstat import flesch_reading_ease, text_standard
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import logging
##
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import subprocess

def ensure_model_downloaded():
    model_path = "./accent-id-commonaccent_ecapa"
    if not os.path.exists(model_path):
        print("Model not found. Cloning from Hugging Face...")
        subprocess.run([
            "git", "clone", 
            "https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa", 
            model_path
        ], check=True)
    else:
        print("Model already exists.")
#ensure_model_downloaded()
# Disable excessive logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
# Configuration
MAX_DURATION = 300  # 5 minutes max for analysis

# Initialize models (cached)
@st.cache_resource
def load_models():
    """Load Whisper and Accent Classifier models"""
    logger.info("Loading models...")
    try:
        # Load Whisper model for transcription
        whisper_model = whisper.load_model("base")
        
        # Load accent classifier
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
        )
        logger.info("Models loaded successfully")
        return whisper_model, classifier
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None

def download_loom_audio_as_wav(loom_url):
    """Download Loom video and extract audio as WAV"""
    logger.info(f"Processing Loom URL: {loom_url}")
    try:
        # Extract video ID
        video_id = loom_url.split("/")[-1].split("?")[0]
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Get video URL from Loom API
        api_url = f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-url"
        response = requests.post(api_url)
        response.raise_for_status()
        video_url = response.json()["url"]
        
        # Download video
        video_path = os.path.join(temp_dir, f"{video_id}.mp4")
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Convert to audio
        audio_path = os.path.join(temp_dir, f"{video_id}.wav")
        audio = AudioSegment.from_file(video_path)
        
        # Trim to MAX_DURATION
        if len(audio) > MAX_DURATION * 1000:
            audio = audio[:MAX_DURATION * 1000]
        
        audio.export(audio_path, format="wav")
        logger.info(f"Audio extracted: {audio_path}")
        
        # Cleanup video file
        os.remove(video_path)
        
        return audio_path, len(audio) / 1000  # Return path and duration in seconds
    except Exception as e:
        logger.error(f"Error processing Loom video: {str(e)}")
        return None, 0

def transcribe_audio_and_analyze_stats(wav_file, whisper_model):
    """Transcribe audio and analyze reading statistics"""
    logger.info(f"Transcribing audio: {wav_file}")
    try:
        # Transcribe audio
        result = whisper_model.transcribe(wav_file)
        text = result['text']
        
        # Calculate timing
        start_time = result['segments'][0]['start'] if result['segments'] else 0
        end_time = result['segments'][-1]['end'] if result['segments'] else 0
        duration = max(end_time - start_time, 1)  # Avoid division by zero
        
        # Calculate statistics
        word_count = len(text.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        # Text readability metrics
        if text.strip():
            fluency_score = flesch_reading_ease(text)
            grade = text_standard(text, float_output=True)
        else:
            fluency_score = 0
            grade = 0
        
        logger.info("Transcription completed")
        return text, fluency_score, grade, wpm, duration
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return "", 0, 0, 0, 0

def analyze_accent(wav_file, classifier):
    """Analyze accent from audio file"""
    logger.info(f"Analyzing accent: {wav_file}")
    try:
        # Load audio file
        signal, fs = torchaudio.load(wav_file)
        
        # Classify accent
        out_prob, score, index, text_lab = classifier.classify_batch(signal)
        
        # Get the top prediction
        accent_label = text_lab[0]
        accent_score = score[0].item()
        
        logger.info(f"Accent detected: {accent_label} ({accent_score:.2f})")
        return accent_label, accent_score
    except Exception as e:
        logger.error(f"Error in accent analysis: {str(e)}")
        return "Unknown", 0

def generate_report(transcript, fluency, grade, wpm, accent_label, accent_score, duration):
    """Generate comprehensive analysis report"""
    # Create accent mapping
    accent_map = {
        "us": "American 🇺🇸",
        "england": "British 🇬🇧",
        "australia": "Australian 🇦🇺",
        "canada": "Canadian 🍁",
        "indian": "Indian 🇮🇳",
        "african": "African 🌍",
        "philippines": "Filipino 🇵🇭"
    }
    
    # Get accent display name
    accent_display = accent_map.get(accent_label.lower(), accent_label.capitalize())
    
    # Fluency interpretation
    if fluency > 90:
        fluency_desc = "Very Easy"
        fluency_emoji = "🌟"
    elif fluency > 80:
        fluency_desc = "Easy"
        fluency_emoji = "👍"
    elif fluency > 70:
        fluency_desc = "Fairly Easy"
        fluency_emoji = "✅"
    elif fluency > 60:
        fluency_desc = "Standard"
        fluency_emoji = "📝"
    elif fluency > 50:
        fluency_desc = "Fairly Difficult"
        fluency_emoji = "⚠️"
    else:
        fluency_desc = "Difficult"
        fluency_emoji = "❌"
    
    # WPM interpretation
    if wpm < 120:
        wpm_desc = "Slow"
        wpm_emoji = "🐢"
    elif wpm < 160:
        wpm_desc = "Moderate"
        wpm_emoji = "🚶"
    else:
        wpm_desc = "Fast"
        wpm_emoji = "🏃"
    
    # Confidence interpretation
    if accent_score > 0.8:
        conf_desc = "High Confidence"
        conf_emoji = "💯"
    elif accent_score > 0.6:
        conf_desc = "Moderate Confidence"
        conf_emoji = "👍"
    else:
        conf_desc = "Low Confidence"
        conf_emoji = "⚠️"
    
    # Generate report
    report = f"""
    ## 🎙️ Comprehensive Audio Analysis Report
    
    **Accent**: {accent_display} {conf_emoji}  
    **Accent Confidence**: {accent_score*100:.1f}% ({conf_desc})  
    **Speech Rate**: {wpm:.1f} words per minute {wpm_emoji} ({wpm_desc})  
    **Readability Score**: {fluency:.1f} {fluency_emoji} ({fluency_desc})  
    **Estimated Grade Level**: {grade:.1f}
    
    ### Key Insights:
    
    **Accent Analysis**:
    - The speaker's accent is identified as **{accent_display}** with {conf_desc.lower()}.
    - This accent is characteristic of speakers from {accent_display.split(' ')[0]}.
    
    **Fluency & Clarity**:
    - The speech has a readability score of **{fluency:.1f}**, which is considered **{fluency_desc.lower()}** to understand.
    - The estimated education level required to understand the speech is **grade {grade:.1f}**.
    
    **Speech Patterns**:
    - The speaker maintains a **{wpm_desc.lower()}** pace of **{wpm:.1f} words per minute**.
    - This speech rate is {"ideal for clear communication" if 120 <= wpm <= 160 
        else "could be too slow for efficient communication" if wpm < 120 
        else "may be challenging for some listeners to follow"}.
    
    **Content Summary**:
    - The audio duration was {duration:.1f} seconds with {len(transcript.split())} words.
    - Key topics discussed: {extract_key_topics(transcript)}
    
    ### Transcription Preview:
    {transcript[:500]}{'...' if len(transcript) > 500 else ''}
    """
    
    return report

def extract_key_topics(text, num_topics=3):
    """Extract key topics from transcript (simplified)"""
    words = text.lower().split()
    if not words:
        return "None detected"
    
    # Simple approach: most frequent nouns
    important_words = [word for word in words if len(word) > 4][:num_topics]
    return ", ".join(set(important_words)) if important_words else "General discussion"

def plot_metrics(fluency, wpm, accent_score):
    """Create visualizations for metrics"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fluency gauge
    ax[0].barh(['Readability'], [fluency], color='skyblue')
    ax[0].set_xlim(0, 100)
    ax[0].set_title('Readability Score')
    ax[0].axvline(x=60, color='red', linestyle='--')
    
    # WPM gauge
    ax[1].barh(['Speech Rate'], [wpm], color='lightgreen')
    ax[1].set_xlim(0, 250)
    ax[1].set_title('Words per Minute')
    ax[1].axvline(x=120, color='green', linestyle='--')
    ax[1].axvline(x=160, color='green', linestyle='--')
    
    # Confidence gauge
    ax[2].barh(['Accent Confidence'], [accent_score*100], color='salmon')
    ax[2].set_xlim(0, 100)
    ax[2].set_title('Accent Confidence')
    ax[2].axvline(x=60, color='red', linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="Loom Audio Analyzer",
        layout="wide",
        page_icon="🎤",
        initial_sidebar_state="expanded"
    )
    
    # Custom styling
    st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stTextInput input, .stFileUploader label {
        background-color: #2c5364 !important;
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6, .st-b7, .st-c0 {
        color: #f8f9fa !important;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50 !important;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-cb {
        background-color: #2c5364;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    .stAlert {
        background-color: #2c5364;
        border-left: 4px solid #4CAF50;
    }
    .metric-box {
        background-color: #2c5364;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎤 Loom Audio Analysis Tool")
    st.markdown("Analyze accent, fluency, and speech patterns from Loom videos")
    
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        1. **Enter a Loom video URL**
        2. We download and extract the audio
        3. Analyze accent using SpeechBrain
        4. Transcribe with Whisper
        5. Calculate speech metrics
        6. Generate comprehensive report
        """)
        
        st.markdown("### Supported Features:")
        st.markdown("""
        - Accent identification
        - Transcription
        - Readability scoring
        - Speech rate analysis
        - Educational level estimation
        """)
        
        st.markdown("### Example Loom URL:")
        st.markdown("`https://www.loom.com/share/1234567890abcdef`")
        
        st.markdown("### Note:")
        st.markdown("Currently only Loom URLs are functional. Other methods are placeholders.")
    
    # Input methods
    input_method = st.radio("Select input method:", 
                           ["Loom URL", "YouTube URL", "Direct Upload"],
                           horizontal=True)
    
    loom_url = ""
    if input_method == "Loom URL":
        loom_url = st.text_input("Enter Loom Video URL:", 
                                placeholder="https://www.loom.com/share/...")
    elif input_method == "YouTube URL":
        st.warning("YouTube analysis is not yet implemented")
        youtube_url = st.text_input("Enter YouTube URL:", disabled=True)
    else:
        st.warning("File upload is not yet implemented")
        uploaded_file = st.file_uploader("Upload MP3/MP4 file", type=["mp3", "mp4"], disabled=True)
    
    # Process button
    if st.button("Analyze Audio", type="primary") and loom_url:
        start_time = datetime.now()
        
        # Load models
        with st.spinner("Loading analysis models..."):
            whisper_model, accent_classifier = load_models()
            if whisper_model is None or accent_classifier is None:
                st.error("Failed to load analysis models. Please try again.")
                return
        
        # Process Loom URL
        with st.spinner("Downloading and processing Loom video..."):
            audio_path, duration = download_loom_audio_as_wav(loom_url)
            if not audio_path:
                st.error("Failed to process Loom video. Please check the URL and try again.")
                return
            
            # Display audio player
            st.audio(audio_path, format="audio/wav")
            st.caption(f"Audio duration: {duration:.1f} seconds")
        
        # Analyze accent
        with st.spinner("Analyzing accent..."):
            accent_label, accent_score = analyze_accent(audio_path, accent_classifier)
        
        # Transcribe and analyze
        with st.spinner("Transcribing and analyzing speech..."):
            transcript, fluency, grade, wpm, trans_duration = transcribe_audio_and_analyze_stats(audio_path, whisper_model)
        
        # Generate report
        with st.spinner("Generating report..."):
            report = generate_report(transcript, fluency, grade, wpm, accent_label, accent_score, duration)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accent", accent_label, f"{accent_score*100:.1f}%")
            col2.metric("Speech Rate", f"{wpm:.1f} WPM")
            col3.metric("Readability", f"{fluency:.1f}")
            col4.metric("Grade Level", f"{grade:.1f}")
            
            # Show visualizations
            st.pyplot(plot_metrics(fluency, wpm, accent_score))
            
            # Display report
            st.markdown(report, unsafe_allow_html=True)
            
            # Show full transcription
            with st.expander("View Full Transcription"):
                st.text_area("Transcription", transcript, height=300)
        
        # Cleanup
        try:
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
        except Exception as e:
            logger.warning(f"Error cleaning up files: {str(e)}")
        
        # Performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        st.success(f"✅ Analysis completed in {processing_time:.1f} seconds!")
        st.balloons()

if __name__ == "__main__":
    main()
