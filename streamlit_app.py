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
from pathlib import Path
from typing import List, Dict, Tuple
import io

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import librosa
import librosa.display

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


def create_spectrogram_with_boxes(
    pcen_data: np.ndarray,
    detections: List[Dict],
    clip_start_time: float,
    clip_end_time: float,
    settings: dict,
    colormap: str = 'inferno',
    vmin: float = 0.0,
    vmax: float = 100.0
) -> Image.Image:
    """
    Create a PCEN spectrogram image with bounding boxes for detections.
    
    Args:
        pcen_data: PCEN features (frequency x time)
        detections: List of detections that overlap with this clip
        clip_start_time: Start time of the clip in seconds
        clip_end_time: End time of the clip in seconds
        settings: PCEN settings dictionary
        colormap: Matplotlib colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        PIL Image with spectrogram and bounding boxes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display spectrogram
    img = librosa.display.specshow(
        pcen_data,
        sr=settings["sr"],
        hop_length=settings["hop_length"],
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    
    # Add colorbar
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Set frequency limits (same as training)
    ax.set_ylim(50, 15000)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'PCEN Spectrogram ({clip_start_time:.1f}s - {clip_end_time:.1f}s)')
    
    # Add bounding boxes for detections in this clip
    for det in detections:
        # Check if detection overlaps with this clip
        if det['time_end'] >= clip_start_time and det['time_start'] <= clip_end_time:
            # Calculate box position relative to clip
            time_start = max(0, det['time_start'] - clip_start_time)
            time_end = min(clip_end_time - clip_start_time, det['time_end'] - clip_start_time)
            
            # Create rectangle (time x frequency)
            rect = patches.Rectangle(
                (time_start, det['freq_low_hz']),
                time_end - time_start,
                det['freq_high_hz'] - det['freq_low_hz'],
                linewidth=2,
                edgecolor=get_species_color(det['species_id']),
                facecolor='none',
                label=f"{det['species']} ({det['confidence']:.2f})"
            )
            ax.add_patch(rect)
            
            # Add species label
            ax.text(
                time_start,
                det['freq_high_hz'],
                f"{det['species']}\n{det['confidence']:.2f}",
                color='white',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=get_species_color(det['species_id']), alpha=0.7),
                verticalalignment='bottom'
            )
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    
    return img_pil


def get_species_color(species_id: int) -> str:
    """Get color for a species from config."""
    if species_id in config.BIRD_COLORS:
        rgb = config.BIRD_COLORS[species_id]
        # Convert RGB to hex
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    else:
        return '#FFFFFF'  # White as default


def create_full_spectrogram_visualization(
    audio: np.ndarray,
    sr: int,
    detections: List[Dict]
) -> Tuple[List[Image.Image], List[Dict]]:
    """
    Create full spectrogram visualization with bounding boxes.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        detections: List of all detections
        
    Returns:
        Tuple of (list of PIL Images, list of clip info dicts)
    """
    # Get PCEN settings
    settings = pcen_inference.get_fft_and_pcen_settings()
    
    # Process audio to clips
    clips, _ = pcen_inference.compute_pcen_for_inference(
        audio,
        sr,
        segment_length_seconds=config.PCEN_SEGMENT_LENGTH
    )
    
    images = []
    clip_info = []
    
    for clip_data in clips:
        # Create spectrogram with boxes
        img = create_spectrogram_with_boxes(
            clip_data['pcen'],
            detections,
            clip_data['start_time'],
            clip_data['end_time'],
            settings
        )
        images.append(img)
        clip_info.append({
            'start_time': clip_data['start_time'],
            'end_time': clip_data['end_time']
        })
    
    return images, clip_info


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
    
    # Lossy format warning
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext in ['.mp3', '.ogg']:
            st.warning("⚠️ Lossy format detected. Use WAV/FLAC for best results.")
    
    # Process button
    if uploaded_file is not None:
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
                    st.info("Loading audio file...")
                    audio, sr = detector.load_audio(tmp_audio_path)
                    duration = len(audio) / sr
                    
                    # Display audio player
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    st.audio(uploaded_file, format=f'audio/{file_ext[1:]}')
                    
                    # Run detection
                    st.info(f"Running detection on {duration:.2f} seconds of audio...")
                    detections = detector.detect_single_file(tmp_audio_path)
                    
                    # Store results in session state
                    st.session_state['detections'] = detections
                    st.session_state['audio'] = audio
                    st.session_state['sr'] = sr
                    st.session_state['detector'] = detector
                    st.session_state['tmp_audio_path'] = tmp_audio_path
                    
                    st.success(f"✅ Detection complete! Found {len(detections)} bird call segments.")
                    
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
        
        st.markdown("---")
        st.header("📈 Detection Results")
        
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
        
        # Spectrograms with bounding boxes
        st.markdown("---")
        st.subheader("🎵 PCEN Spectrograms with Detections")
        
        with st.spinner("Generating spectrograms with bounding boxes..."):
            images, clip_info = create_full_spectrogram_visualization(audio, sr, detections)
        
        # Display spectrograms
        if images:
            # Add clip selector
            clip_index = st.slider(
                "Select Clip",
                min_value=0,
                max_value=len(images) - 1,
                value=0,
                format=f"Clip %d / {len(images)}"
            )
            
            st.write(f"**Time range:** {clip_info[clip_index]['start_time']:.1f}s - {clip_info[clip_index]['end_time']:.1f}s")
            st.image(images[clip_index], width='stretch')
            
            # Show detections in this clip
            clip_start = clip_info[clip_index]['start_time']
            clip_end = clip_info[clip_index]['end_time']
            clip_detections = [
                d for d in detections
                if d['time_end'] >= clip_start and d['time_start'] <= clip_end
            ]
            
            if clip_detections:
                st.write(f"**Detections in this clip:** {len(clip_detections)}")
            else:
                st.write("No detections in this clip")
        
        # Detection table
        st.markdown("---")
        st.subheader("📋 Detailed Detection Table")
        
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

