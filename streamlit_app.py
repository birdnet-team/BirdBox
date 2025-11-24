#!/usr/bin/env python3
"""
Streamlit app for bird call detection using trained YOLO models.

This app allows users to upload audio files, select models, adjust detection parameters,
and visualize detections with PCEN spectrograms and bounding boxes.
"""

import os
import sys
import tempfile
import json
import base64
import random
from pathlib import Path
from typing import List, Dict
import io

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import librosa

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from inference.detect_birds import BirdCallDetector
from inference.utils import pcen_inference


# Default model URL for download if no models found
DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
DEFAULT_MODEL_NAME = "yolov8n.pt"


def find_available_models(models_dir: Path) -> List[str]:
    """Find all available model files in the models directory."""
    if not models_dir.exists():
        return []
    
    model_extensions = ['.pt', '.onnx', '.engine']
    models = []
    
    for ext in model_extensions:
        models.extend([str(f) for f in models_dir.glob(f'*{ext}')])
    
    return sorted(models)


def download_default_model(models_dir: Path) -> str:
    """Download a default model if none are available."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / DEFAULT_MODEL_NAME
    
    if not model_path.exists():
        st.info(f"Downloading default model to {model_path}...")
        try:
            from ultralytics import YOLO
            # This will download the model
            YOLO(DEFAULT_MODEL_NAME)
            # Move it to models directory
            default_location = Path.home() / '.cache' / 'ultralytics' / DEFAULT_MODEL_NAME
            if default_location.exists():
                import shutil
                shutil.copy(default_location, model_path)
            st.success("Default model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download default model: {e}")
            return None
    
    return str(model_path)


def get_species_color(species_id: int) -> str:
    """Get color for a species from config."""
    if species_id in config.BIRD_COLORS:
        rgb = config.BIRD_COLORS[species_id]
        # Convert RGB to hex
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    else:
        return '#FFFFFF'  # White as default


def hz_to_mel_normalized(freq_hz: float, min_freq: float = 50.0, max_freq: float = 15000.0) -> float:
    """
    Convert frequency in Hz to normalized mel value [0, 1].
    This is the reverse of pixels_to_hz in detect_birds.py.
    
    Args:
        freq_hz: Frequency in Hz
        min_freq: Minimum frequency (default 50 Hz)
        max_freq: Maximum frequency (default 15000 Hz)
        
    Returns:
        Normalized mel value in [0, 1]
    """
    # Convert Hz to mel using HTK scale (same as training)
    mel_value = librosa.hz_to_mel(freq_hz, htk=True)
    
    # Calculate mel range
    min_mel = librosa.hz_to_mel(min_freq, htk=True)
    max_mel = librosa.hz_to_mel(max_freq, htk=True)
    mel_range = max_mel - min_mel
    
    # Normalize to [0, 1]
    mel_normalized = (mel_value - min_mel) / mel_range
    
    return mel_normalized


def create_full_spectrogram_visualization(
    audio: np.ndarray,
    sr: int,
    detections: List[Dict],
    colormap: str = 'inferno',
    vmin: float = 0.0,
    vmax: float = 100.0
) -> Image.Image:
    """
    Create a simple, wide spectrogram image for horizontal scrolling (no axes or labels).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        detections: List of all detections
        colormap: Matplotlib colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        PIL Image with simple spectrogram and bounding boxes (no axes)
    """
    # Get PCEN settings
    settings = pcen_inference.get_fft_and_pcen_settings()
    target_sr = settings["sr"]  # 32000 Hz
    hop_length = settings["hop_length"]
    
    # Validate audio
    librosa.util.valid_audio(audio)
    
    # Map to the range [-2**31, 2**31[ (same as training)
    audio = (audio * (2 ** 31)).astype("float32")
    
    # Resample if needed (same as in training)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Pre-pad with ~0.5s of repeated audio (same as training)
    pad_len = int(settings["left_pad_length"] * sr)
    audio_padded = np.concatenate([audio[:pad_len], audio])
    
    # Compute Short-Term Fourier Transform (STFT) - same as training
    stft = librosa.stft(
        audio_padded,
        n_fft=settings["n_fft"],
        win_length=settings["win_length"],
        hop_length=hop_length,
        window=settings["window"],
        center=False,
    )
    
    # Compute squared magnitude coefficients
    abs2_stft = np.abs(stft) ** 2
    del stft  # Free memory
    
    # Gather frequency bins according to the Mel scale (same as training)
    melspec = librosa.feature.melspectrogram(
        S=abs2_stft,
        sr=sr,
        n_fft=settings["n_fft"],
        n_mels=settings["n_mels"],
        fmin=settings["fmin"],
        fmax=settings["fmax"],
        htk=True,
    )
    del abs2_stft  # Free memory
    
    # Loop the spectrogram in time domain to avoid PCEN initialization artifacts (same as training)
    loop_length = min(100, melspec.shape[1] // 4)  # Loop first 25% or 100 frames
    if loop_length > 0:
        melspec_looped = np.concatenate([melspec[:, :loop_length], melspec], axis=1)
        del melspec  # Free memory
    else:
        melspec_looped = melspec
    
    # Compute PCEN (same parameters as training)
    pcen_looped = librosa.pcen(
        melspec_looped,
        sr=sr,
        hop_length=hop_length,
        gain=settings["pcen_norm_exponent"],
        bias=settings["pcen_delta"],
        power=settings["pcen_power"],
        time_constant=settings["pcen_time_constant"],
    )
    del melspec_looped  # Free memory
    
    # Extract the original segment (skip the looped part)
    pcen_segment = pcen_looped[:, loop_length:] if loop_length > 0 else pcen_looped
    del pcen_looped  # Free memory
    
    # Drop padded frames (same as training)
    pad_frames = pad_len // hop_length
    pcen_data = pcen_segment[:, pad_frames:].astype("float32")
    del pcen_segment  # Free memory
    
    # Get spectrogram dimensions
    n_mels, n_time = pcen_data.shape
    
    # Calculate actual duration based on the audio length (before padding)
    duration = len(audio) / sr
    
    # Create figure - wide for scrolling, without axes
    # Calculate pixels per second for good resolution
    pixels_per_second = 100  # 100 pixels per second gives good detail
    width_pixels = int(duration * pixels_per_second)
    height_pixels = 600  # Fixed height
    
    # Calculate figure size in inches (dpi will be 100)
    dpi = 100
    fig_width = width_pixels / dpi
    fig_height = height_pixels / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # Full figure, no margins
    ax.set_axis_off()  # No axes
    fig.add_axes(ax)
    
    # Display spectrogram without axes
    # Use extent in normalized mel coordinates [0, 1] for y-axis
    img = ax.imshow(
        pcen_data,
        aspect='auto',
        origin='lower',
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, duration, 0, 1]  # time in seconds (matching detections), normalized mel [0, 1]
    )
    
    # Add bounding boxes for all detections
    for det in detections:
        # Convert Hz to normalized mel coordinates
        freq_low_norm = hz_to_mel_normalized(det['freq_low_hz'])
        freq_high_norm = hz_to_mel_normalized(det['freq_high_hz'])
        
        # Create rectangle (time x normalized mel)
        rect = patches.Rectangle(
            (det['time_start'], freq_low_norm),
            det['time_end'] - det['time_start'],
            freq_high_norm - freq_low_norm,
            linewidth=2,
            edgecolor=get_species_color(det['species_id']),
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add species label
        label_offset = 0.02  # Small offset in normalized coordinates
        ax.text(
            det['time_start'],
            freq_high_norm + label_offset,
            f"{det['species']} {det['confidence']:.2f}",
            color='white',
            fontsize=8,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=get_species_color(det['species_id']), alpha=0.8),
            verticalalignment='bottom'
        )
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    
    return img_pil


def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable objects to standard Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def format_detections_for_table(detections: List[Dict]) -> pd.DataFrame:
    """Format detections as a pandas DataFrame for display."""
    if not detections:
        return pd.DataFrame()
    
    # Create DataFrame with relevant columns
    df_data = []
    for i, det in enumerate(detections, 1):
        row = {
            '#': i,
            'Species': det['species'],
            'Confidence': f"{det['confidence']:.3f}",
            'Start (s)': f"{det['time_start']:.2f}",
            'End (s)': f"{det['time_end']:.2f}",
            'Duration (s)': f"{det['time_end'] - det['time_start']:.2f}",
            'Freq Low (Hz)': det['freq_low_hz'],
            'Freq High (Hz)': det['freq_high_hz'],
        }
        
        # Add merged info if available
        if 'detections_merged' in det:
            row['Clips Merged'] = det['detections_merged']
            row['Max Confidence'] = f"{det['max_confidence']:.3f}"
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def main():
    st.set_page_config(
        page_title="BirdBox - Bird Call Detection",
        layout="wide",
        page_icon="🐦"
    )
    
    st.title("BirdBox - Bird Call Detection")
    st.markdown("Upload audio files to detect bird calls using trained YOLO models")
    
    # Sidebar with logo - using base64 encoding to bypass media server issues
    logo_path = "img/logo_birdbox.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{logo_base64}" style="width: 100%; margin-top: -30px;">',
            unsafe_allow_html=True
        )
    
    st.sidebar.header("Settings")
    
    # Model selection
    models_dir = Path(__file__).parent / "models"
    available_models = find_available_models(models_dir)
    
    if not available_models:
        st.sidebar.warning("No models found in models directory")
        if st.sidebar.button("Download Default Model"):
            default_model = download_default_model(models_dir)
            if default_model:
                available_models = [default_model]
                st.rerun()
    
    if available_models:
        # Display model names without full path
        model_names = [Path(m).name for m in available_models]
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            model_names,
            help="Choose a trained model for bird call detection"
        )
        selected_model = available_models[model_names.index(selected_model_name)]
    else:
        st.error("No models available. Please add models to the models directory or download a default model.")
        st.stop()
    
    # Detection parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Parameters")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        format="%.2f",
        help="Minimum confidence score for detections (lower = more detections)"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression"
    )
    
    song_gap_threshold = st.sidebar.slider(
        "Song Gap Threshold (s)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="Maximum gap between detections to merge into same song"
    )
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Current Dataset:** {config.DATASET_NAME}")
    st.sidebar.info(f"**Species Count:** {len(config.ID_TO_EBIRD_CODES)}")
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'flac', 'ogg', 'mp3'],
        help="Supported formats: WAV, FLAC, OGG, MP3 (WAV or FLAC recommended for best results)",
        label_visibility="collapsed"
    )
    
    # Check if a new file was uploaded and clear previous results
    if uploaded_file is not None:
        current_filename = uploaded_file.name
        if 'uploaded_filename' in st.session_state and st.session_state['uploaded_filename'] != current_filename:
            # Clear all detection results when a new file is uploaded
            for key in ['detections', 'audio', 'sr', 'detector', 'tmp_audio_path', 'uploaded_filename', 'just_completed']:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Store current filename
        st.session_state['uploaded_filename'] = current_filename
    
    # Lossy format warning
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext in ['.mp3', '.ogg']:
            st.warning("⚠️ Lossy format detected. Use WAV/FLAC for best results.")
    
    # Process button (hide if results already exist)
    if uploaded_file is not None and 'detections' not in st.session_state:
        if st.button("Detect Bird Calls", type="primary"):
            with st.spinner("Processing audio file..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_audio_path = tmp_file.name
                    
                    # Initialize detector
                    detector = BirdCallDetector(
                        model_path=selected_model,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        song_gap_threshold=song_gap_threshold
                    )
                    
                    # Load audio
                    # st.info("Loading audio file...")
                    audio, sr = detector.load_audio(tmp_audio_path)
                    duration = len(audio) / sr
                    
                    # Run detection with progress bar
                    # st.info(f"Running detection on {duration:.2f} seconds of audio...")
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def update_progress(current, total, message):
                        """Update Streamlit progress bar"""
                        progress = current / total
                        progress_bar.progress(progress)
                        progress_text.text(f"{message} ({current}/{total} clips)")
                    
                    detections = detector.detect_single_file(tmp_audio_path, progress_callback=update_progress)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    progress_text.empty()
                    
                    # Store results in session state
                    st.session_state['detections'] = detections
                    st.session_state['audio'] = audio
                    st.session_state['sr'] = sr
                    st.session_state['detector'] = detector
                    st.session_state['tmp_audio_path'] = tmp_audio_path
                    st.session_state['just_completed'] = True
                    
                    # Rerun to hide the button and show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results if available
    if 'detections' in st.session_state:
        detections = st.session_state['detections']
        audio = st.session_state['audio']
        sr = st.session_state['sr']
        detector = st.session_state['detector']
        
        # Show success message if just completed
        if st.session_state.get('just_completed', False):
            st.success(f"Detection complete! Found {len(detections)} bird call segments.")
            st.session_state['just_completed'] = False
        
        st.markdown("---")
        # st.header("Detection Results")
        
        # PCEN Spectrogram with Detections (shown first)
        st.subheader("PCEN Spectrogram with Detections")
        
        duration = len(audio) / sr
        st.write(f"**Audio duration:** {duration:.1f}s | **Detections:** {len(detections)} | Scroll horizontally to navigate through the audio timeline")
        
        with st.spinner("Generating and rendering the spectrogram with included bounding boxes. This may take a while."):
            full_spectrogram = create_full_spectrogram_visualization(audio, sr, detections)
        
        # Display spectrogram in scrollable container
        if full_spectrogram:
            # Convert image to base64 for HTML display
            buf = io.BytesIO()
            full_spectrogram.save(buf, format='PNG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            
            # Create horizontally scrollable container with mouse wheel scrolling
            components.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        * {{
                            margin: 0;
                            padding: 0;
                            box-sizing: border-box;
                        }}
                        body {{
                            margin: 0;
                            padding: 0;
                            overflow: hidden;
                            background-color: #000;
                        }}
                        #border-wrapper {{
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            background-color: #000;
                            width: 100%;
                            height: 100%;
                            overflow: hidden;
                            box-sizing: border-box;
                        }}
                        #spectrogram-container {{
                            overflow-x: auto;
                            overflow-y: hidden;
                            background-color: #000;
                            width: 100%;
                            height: 100%;
                        }}
                        #spectrogram-container img {{
                            height: 600px;
                            width: auto;
                            display: block;
                            vertical-align: top;
                        }}
                    </style>
                </head>
                <body>
                    <div id="border-wrapper">
                        <div id="spectrogram-container">
                            <img src="data:image/png;base64,{img_base64}" alt="Spectrogram">
                        </div>
                    </div>
                    <script>
                        const container = document.getElementById('spectrogram-container');
                        container.addEventListener('wheel', function(e) {{
                            if (Math.abs(e.deltaY) > 0) {{
                                e.preventDefault();
                                container.scrollLeft += e.deltaY;
                            }}
                        }}, {{ passive: false }});
                    </script>
                </body>
                </html>
                """,
                height=622,
                scrolling=False
            )
            # st.caption("Scroll horizontally to navigate through the audio timeline")
        
        # Vertical spacer (adjust height value to customize spacing)
        # st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        
        # Audio player
        if uploaded_file is not None:
            file_ext = Path(uploaded_file.name).suffix.lower()
            st.audio(uploaded_file, format=f'audio/{file_ext[1:]}')
        
        st.markdown("---")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(detections))
        
        with col2:
            unique_species = len(set(d['species'] for d in detections))
            st.metric("Species Detected", unique_species)
        
        with col3:
            if detections:
                avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        with col4:
            if detections:
                total_duration = sum(d['time_end'] - d['time_start'] for d in detections)
                st.metric("Total Duration", f"{total_duration:.1f}s")
        
        # Species breakdown
        if detections:
            st.subheader("Species Breakdown")
            species_counts = {}
            for det in detections:
                species = det['species']
                if species not in species_counts:
                    species_counts[species] = 0
                species_counts[species] += 1
            
            species_df = pd.DataFrame([
                {'Species': species, 'Count': count}
                for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(species_df, width='stretch', hide_index=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                species_df_plot = species_df.head(10)  # Top 10 species
                ax.barh(species_df_plot['Species'], species_df_plot['Count'])
                ax.set_xlabel('Count')
                ax.set_title('Top Species Detected')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
        
        # Detection table
        st.markdown("---")
        st.subheader("Detailed Detection Table")
        
        df = format_detections_for_table(detections)
        if not df.empty:
            st.dataframe(df, width='stretch', hide_index=True, height=400)
        
        # Download section
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            json_data = {
                'audio_file': uploaded_file.name if 'uploaded_file' in locals() else 'unknown',
                'model_config': {
                    'model': str(selected_model),
                    'confidence_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'song_gap_threshold': song_gap_threshold,
                    'dataset': config.DATASET_NAME,
                },
                'detection_count': len(detections),
                'detections': detections
            }
            
            # Convert numpy types to JSON-serializable types
            json_data = convert_to_json_serializable(json_data)
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{Path(uploaded_file.name).stem}_detections.json",
                mime="application/json"
            )
        
        with col2:
            # CSV download
            csv_data = []
            for det in detections:
                csv_data.append({
                    'Filename': uploaded_file.name if 'uploaded_file' in locals() else 'unknown',
                    'Start Time (s)': f"{det['time_start']:.1f}",
                    'End Time (s)': f"{det['time_end']:.1f}",
                    'Low Freq (Hz)': det['freq_low_hz'],
                    'High Freq (Hz)': det['freq_high_hz'],
                    'Species eBird Code': det['species'],
                    'Confidence': f"{det['confidence']:.3f}"
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_str = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_str,
                file_name=f"{Path(uploaded_file.name).stem}_detections.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>BirdBox - Bird Call Detection System</p>
            <p style='font-size: 0.8em; color: gray;'>
                Upload audio files in WAV, FLAC, OGG, or MP3 format. 
                Adjust detection parameters in the sidebar for optimal results.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

