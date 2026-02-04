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
import time
import threading
from pathlib import Path
from typing import List, Dict
import io

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend BEFORE importing pyplot to prevent GUI conflicts in multi-user scenarios
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import librosa
import soundfile as sf

# Global lock for thread-safe matplotlib figure creation
# Matplotlib is not thread-safe when multiple users create figures simultaneously
_matplotlib_lock = threading.Lock()

# Add src directory to path (go up one level from src/streamlit/app.py to src/)
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add streamlit directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from inference.detect_birds import BirdCallDetector
from inference.utils import pcen_inference
from concurrency_manager import get_concurrency_manager, ConcurrencyConfig


# Default model URL for download if no models found
# For Nextcloud/TUC Cloud folder shares, use the WebDAV ZIP download endpoint:
DEFAULT_MODEL_URL = "https://tuc.cloud/public.php/dav/files/HcbKnxFsfHYyq5G/?accept=zip"
DEFAULT_MODEL_NAME = "Hawaii.pt"  # Name of the model file inside the ZIP archive


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
    """Download a default model from the specified URL if none are available.
    
    Supports both direct file downloads and ZIP archive downloads (for Nextcloud/TUC Cloud folder shares).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / DEFAULT_MODEL_NAME
    
    if not model_path.exists():
        st.info(f"Downloading default model from {DEFAULT_MODEL_URL}...")
        try:
            import urllib.request
            import zipfile
            import tempfile
            from urllib.error import URLError, HTTPError
            
            # Check if URL is a ZIP file (Nextcloud folder download)
            is_zip = DEFAULT_MODEL_URL.endswith('?accept=zip') or DEFAULT_MODEL_URL.endswith('.zip')
            
            if is_zip:
                # Download ZIP archive to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    tmp_zip_path = tmp_zip.name
                
                try:
                    # Download ZIP
                    req = urllib.request.Request(DEFAULT_MODEL_URL)
                    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                    
                    with urllib.request.urlopen(req) as response:
                        with open(tmp_zip_path, 'wb') as f:
                            f.write(response.read())
                    
                    # Extract ZIP and find the model file
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                        # List all files in the ZIP
                        file_list = zip_ref.namelist()
                        
                        # Look for the model file (could be in a subdirectory)
                        model_found = False
                        for file_path in file_list:
                            # Check if this file matches our model name (could be in subdirectory)
                            if file_path.endswith(DEFAULT_MODEL_NAME) or Path(file_path).name == DEFAULT_MODEL_NAME:
                                # Extract the model file
                                zip_ref.extract(file_path, models_dir)
                                
                                # If it was in a subdirectory, move it to the models directory root
                                extracted_path = models_dir / file_path
                                if extracted_path != model_path:
                                    import shutil
                                    shutil.move(str(extracted_path), str(model_path))
                                
                                model_found = True
                                break
                        
                        if not model_found:
                            st.error(f"Model file '{DEFAULT_MODEL_NAME}' not found in ZIP archive.")
                            st.info(f"Files in ZIP: {', '.join(file_list[:10])}{'...' if len(file_list) > 10 else ''}")
                            return None
                    
                    st.success(f"Default model extracted successfully to {model_path}!")
                    
                finally:
                    # Clean up temporary ZIP file
                    if os.path.exists(tmp_zip_path):
                        os.unlink(tmp_zip_path)
                        
            else:
                # Direct file download
                req = urllib.request.Request(DEFAULT_MODEL_URL)
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                
                with urllib.request.urlopen(req) as response:
                    with open(model_path, 'wb') as f:
                        f.write(response.read())
                
                st.success(f"Default model downloaded successfully to {model_path}!")
                
        except HTTPError as e:
            st.error(f"Failed to download model: HTTP error {e.code} - {e.reason}")
            return None
        except URLError as e:
            st.error(f"Failed to download model: URL error - {e.reason}")
            return None
        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid ZIP archive.")
            return None
        except Exception as e:
            st.error(f"Failed to download default model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    return str(model_path)


def get_species_color(species_id: int, bird_colors: Dict = None) -> str:
    """
    Get color for a species.
    
    Args:
        species_id: Species ID number
        bird_colors: Dictionary mapping species IDs to RGB colors (if None, uses default Hawaii dataset colors)
    
    Returns:
        Hex color string
    """
    if bird_colors is None:
        # Fallback to default species mapping (Hawaii) if not provided
        default_config = config.get_species_mapping('Hawaii')
        bird_colors = default_config['bird_colors']
    
    if species_id in bird_colors:
        rgb = bird_colors[species_id]
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
    vmax: float = 100.0,
    bird_colors: Dict = None
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
    
    # Validate pcen_data
    if pcen_data.size == 0:
        raise ValueError("PCEN data is empty")
    if np.any(np.isnan(pcen_data)) or np.any(np.isinf(pcen_data)):
        # Replace NaN/Inf with 0
        pcen_data = np.nan_to_num(pcen_data, nan=0.0, posinf=vmax, neginf=vmin)
    
    # Calculate actual duration based on the audio length (before padding)
    duration = len(audio) / sr
    
    # Ensure minimum dimensions
    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {duration}")
    
    # Create figure - wide for scrolling, without axes
    # Calculate pixels per second for good resolution
    pixels_per_second = 100  # 100 pixels per second gives good detail
    width_pixels = max(100, int(duration * pixels_per_second))  # Minimum 100 pixels width
    height_pixels = 600  # Fixed height
    
    # Calculate figure size in inches (dpi will be 100)
    dpi = 100
    fig_width = width_pixels / dpi
    fig_height = height_pixels / dpi
    
    # Use lock to prevent concurrent matplotlib operations from interfering
    # Matplotlib is not thread-safe when multiple users create figures simultaneously
    with _matplotlib_lock:
        fig = None
        img_pil = None
        try:
            # Create new figure (don't use plt.clf() as it can interfere with figure creation)
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
            if fig is None:
                raise RuntimeError("Failed to create matplotlib figure")
            
            ax = fig.add_axes([0., 0., 1., 1.])  # Full figure, no margins
            ax.set_axis_off()  # No axes
            
            # Display spectrogram without axes
            # Use extent in normalized mel coordinates [0, 1] for y-axis
            im = ax.imshow(
                pcen_data,
                aspect='auto',
                origin='lower',
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, 0, 1],  # time in seconds (matching detections), normalized mel [0, 1]
                interpolation='nearest'
            )
            
            # Add bounding boxes for all detections
            for det in detections:
                try:
                    # Convert Hz to normalized mel coordinates
                    freq_low_norm = hz_to_mel_normalized(det['freq_low_hz'])
                    freq_high_norm = hz_to_mel_normalized(det['freq_high_hz'])
                    
                    # Create rectangle (time x normalized mel)
                    rect = patches.Rectangle(
                        (det['time_start'], freq_low_norm),
                        det['time_end'] - det['time_start'],
                        freq_high_norm - freq_low_norm,
                        linewidth=2,
                        edgecolor=get_species_color(det['species_id'], bird_colors),
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add species label
                    label_offset = 0.02  # Small offset in normalized coordinates
                    ax.text(
                        det['time_start'],
                        freq_high_norm + label_offset,
                        f"{det['species']} {(det['avg_confidence'] if 'detections_merged' in det else det['confidence']):.2f}",
                        color='white',
                        fontsize=8,
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=get_species_color(det['species_id'], bird_colors), alpha=0.8),
                        verticalalignment='bottom'
                    )
                except Exception:
                    # Skip individual detection if it causes an error
                    continue
            
            # Verify figure is still valid before saving
            if fig is None or not hasattr(fig, 'canvas') or fig.canvas is None:
                raise RuntimeError("Figure is invalid or has been closed")
            
            # Convert to PIL Image - save BEFORE closing the figure
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='black', edgecolor='none')
            buf.seek(0)
            img_pil = Image.open(buf)
            # Make sure the image is loaded into memory before closing figure
            img_pil.load()
            
        except Exception as e:
            # Log the error but don't return black image - let it propagate so we can see what's wrong
            import traceback
            print(f"Error creating spectrogram: {e}")
            print(traceback.format_exc())
            # Re-raise the exception so the caller can handle it
            raise
        finally:
            # Always close figure to prevent memory leaks and conflicts in multi-user scenarios
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass  # Ignore errors when closing
        
        # Return the image after the figure is closed
        if img_pil is None:
            raise RuntimeError("Failed to create spectrogram image")
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


def get_common_name_from_ebird_code(ebird_code: str, species_mappings: Dict) -> str:
    """
    Extract common name from eBird code using species mappings.
    
    Args:
        ebird_code: eBird species code
        species_mappings: Species mapping dictionary with 'ebird_to_name' key
        
    Returns:
        Common name string, or eBird code if not found
    """
    ebird_to_name = species_mappings.get('ebird_to_name', {})
    full_name = ebird_to_name.get(ebird_code, None)
    
    if full_name and '_' in full_name:
        # Split on first underscore: "Scientific Name_Common Name"
        parts = full_name.split('_', 1)
        if len(parts) > 1:
            return parts[1]  # Return common name
    
    # Fallback to eBird code if name not found or format is unexpected
    return ebird_code


def format_detections_for_table(detections: List[Dict], species_mappings: Dict = None) -> pd.DataFrame:
    """Format detections as a pandas DataFrame for display."""
    if not detections:
        return pd.DataFrame()
    
    # Create DataFrame with relevant columns
    df_data = []
    for i, det in enumerate(detections, 1):
        # Get common name for display if species_mappings provided
        if species_mappings:
            species_display = get_common_name_from_ebird_code(det['species'], species_mappings)
        else:
            species_display = det['species']
        
        row = {
            '#': i,
            'Species': species_display,
            'Confidence': f"{(det['avg_confidence'] if 'detections_merged' in det else det['confidence']):.3f}",
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
    
    # Custom CSS to fix selectbox dropdown highlighting
    st.markdown("""
        <style>
        /* Style for dropdown options - invert default behavior */
        [data-baseweb="select"] ul li:hover {
            background-color: rgb(220, 220, 220) !important;
        }
        [data-baseweb="select"] ul li[aria-selected="true"] {
            background-color: rgb(240, 242, 246) !important;
        }
        [data-baseweb="select"] ul li[aria-selected="true"]:hover {
            background-color: rgb(220, 220, 220) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Calculate project root (go up from src/streamlit/app.py to project root)
    project_root = Path(__file__).parent.parent.parent
    
    # Sidebar with logo - using base64 encoding to bypass media server issues
    logo_path = project_root / "img" / "logo_birdbox.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{logo_base64}" style="width: 100%; max-width: 280px; margin-top: -30px;">',
            unsafe_allow_html=True
        )
    
    st.sidebar.header("Settings")
    
    # Initialize concurrency manager with constants
    # Configure these values directly in the code below
    CONCURRENCY_CONTROL_ENABLED = True  # Set to False to disable concurrency control
    # Note: To change max concurrent detections, modify MAX_CONCURRENT_DETECTIONS
    # in config.py and restart the Streamlit server
    
    if CONCURRENCY_CONTROL_ENABLED:
        # Use default from concurrency_manager.py
        concurrency_config = ConcurrencyConfig()
        
        # Get concurrency manager instance (config only used on first call)
        concurrency_manager = get_concurrency_manager(concurrency_config)
        
        # Get unique session ID for concurrency control
        # Streamlit doesn't provide a stable session ID, so we create one from session state
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state['session_id'] = str(uuid.uuid4())
        session_id = st.session_state['session_id']
    else:
        # Concurrency control disabled - create a dummy manager
        concurrency_manager = None
        session_id = None
        concurrency_config = None
    
    # Model selection (models directory is at project root)
    models_dir = project_root / "models"
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
            help="""
            Choose a trained model for bird call detection.\\
            All models located in the models/ directory are listed here.
            """
        )
        selected_model = available_models[model_names.index(selected_model_name)]
    else:
        st.error("No models available. Please add models to the models directory or download a default model.")
        st.stop()
    
    # Get species mapping for selected model
    try:
        species_mapping_name = config.get_species_mapping_for_model(selected_model)
        species_mappings = config.get_species_mapping(species_mapping_name)
    except ValueError as e:
        st.sidebar.error("⚠️ Could not determine species mapping for selected model")
        st.sidebar.warning(str(e))
        st.stop()
    
    # Store in session state for use throughout the app
    st.session_state['species_mapping'] = species_mapping_name
    st.session_state['species_mappings'] = species_mappings
    
    # Species count and list
    # st.sidebar.info(f"**Species Count:** {len(species_mappings['id_to_ebird'])}")
    
    # Species list section
    with st.sidebar.expander("view species list for the selected model", expanded=False):
        id_to_ebird = species_mappings['id_to_ebird']
        ebird_to_name = species_mappings.get('ebird_to_name', {})
        
        # If ebird_to_name is empty, try to get it directly from config (in case of stale session state)
        if not ebird_to_name:
            try:
                species_mapping_name = st.session_state.get('species_mapping', 'Hawaii')
                fresh_config = config.get_species_mapping(species_mapping_name)
                ebird_to_name = fresh_config.get('ebird_to_name', {})
                # Update session state with fresh data
                st.session_state['species_mappings'] = fresh_config
            except Exception:
                pass
        
        # Create list with eBird code, scientific name, and common name
        species_list = []
        species_codes = sorted(set(id_to_ebird.values()))
        
        for code in species_codes:
            full_name = ebird_to_name.get(code, "Name not available")
            
            # Split full name into scientific and common name (separated by underscore)
            if full_name != "Name not available" and '_' in full_name:
                parts = full_name.split('_', 1)  # Split on first underscore only
                scientific_name = parts[0] if len(parts) > 0 else "Unknown"
                common_name = parts[1] if len(parts) > 1 else "Unknown"
            else:
                scientific_name = full_name
                common_name = "Unknown"
            
            species_list.append({
                'eBird Code': code,
                'Scientific Name': scientific_name,
                'Common Name': common_name
            })
        
        species_df = pd.DataFrame(species_list)
        
        # Display as table
        st.dataframe(
            species_df,
            hide_index=True,
            height=300,
            width='stretch'
        )
        
        # Download buttons
        # CSV download
        csv_str = species_df.to_csv(index=False)
        species_mapping_name = st.session_state.get('species_mapping', 'Hawaii')
        st.download_button(
            label="Download as CSV",
            data=csv_str,
            file_name=f"{species_mapping_name}_species_list.csv",
            mime="text/csv",
            key="download_species_csv",
            use_container_width=True
        )
        
        # JSON download
        json_data = {
            'species_mapping': species_mapping_name,
            'species_count': len(species_codes),
            'species': [
                {
                    'code': row['eBird Code'],
                    'scientific_name': row['Scientific Name'],
                    'common_name': row['Common Name']
                }
                for _, row in species_df.iterrows()
            ]
        }
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"{species_mapping_name}_species_list.json",
            mime="application/json",
            key="download_species_json",
            use_container_width=True
        )
    
    # Detection parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Parameters")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=0.8,
        value=0.18,
        step=0.01,
        format="%.2f",
        help="""
        Minimum confidence score for individual detections.\\
        Decrease to retrieve more detections and therefore emphasize Recall.\\
        Increase to retrieve less detections and therefore emphasize Precision.\\
        Recommended: 0.25
        """
    )
    
    # IoU Threshold - fixed at 0.5 (not user-adjustable)
    # this is the iou threshold for the nms algorithm, not for any validation
    NMS_IOU_THRESHOLD = 0.5
    
    song_gap_threshold = st.sidebar.slider(
        "Song Gap Threshold (seconds)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.01,
        help="""
        Maximum gap between detections to merge into same song.\\
        Decrease to retrieve more individual detections.\\
        Increase to merge more detections into songs.\\
        Recommended: 0.1s for most species, adjust based on species vocalization patterns.
        """
    )
    
    # Check if model or parameters have changed and clear results if they have
    should_reset = False
    
    # Check model change
    if 'previous_model' in st.session_state:
        if st.session_state['previous_model'] != selected_model and 'detections' in st.session_state:
            should_reset = True
    
    # Check confidence threshold change
    if 'previous_conf_threshold' in st.session_state:
        if st.session_state['previous_conf_threshold'] != conf_threshold and 'detections' in st.session_state:
            should_reset = True
    
    # Check song gap threshold change
    if 'previous_song_gap_threshold' in st.session_state:
        if st.session_state['previous_song_gap_threshold'] != song_gap_threshold and 'detections' in st.session_state:
            should_reset = True
    
    # Reset if any parameter changed
    if should_reset:
        for key in ['detections', 'audio', 'sr', 'tmp_audio_path', 'just_completed', 'detection_in_progress', 'model_path']:
            if key in st.session_state:
                del st.session_state[key]
        st.info("Settings changed. Click 'Detect Bird Calls' to run detection with the selected settings.")
    
    # Store current values for next comparison
    st.session_state['previous_model'] = selected_model
    st.session_state['previous_conf_threshold'] = conf_threshold
    st.session_state['previous_song_gap_threshold'] = song_gap_threshold
    
    # Main content area
    # Get max duration for display
    MAX_DURATION_SECONDS = config.MAX_DURATION_SECONDS
    MAX_DURATION_MINUTES = MAX_DURATION_SECONDS / 60
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'flac', 'ogg', 'mp3'],
        help=f"Supported formats: WAV, FLAC, OGG, MP3 (WAV or FLAC recommended for best results). Maximum file length: {MAX_DURATION_MINUTES:.0f} minutes. Longer files will be automatically truncated.",
        label_visibility="collapsed"
    )
    
    # Display max file length info below the uploader
    if uploaded_file is None:
        st.caption(f"""
            Maximum file length for WebApp showcase: **{MAX_DURATION_MINUTES:.0f} minutes**. Files longer than this will be automatically truncated to the first {MAX_DURATION_MINUTES:.0f} minutes.\\
            For huge datasets use the [BirdBox CLI](https://github.com/birdnet-team/BirdBox) instead.    
        """)

    st.info("Note: BirdBox is still under active development. Performance may vary.")

    # Check if file was removed (user clicked X) and clear all results
    if uploaded_file is None and 'uploaded_filename' in st.session_state:
        # Clean up truncated file if it exists
        truncated_path = st.session_state.get('truncated_audio_path')
        if truncated_path is not None and os.path.exists(truncated_path):
            try:
                os.unlink(truncated_path)
            except Exception:
                pass  # Ignore errors during cleanup
        
        # Clear all detection results and queue state when file is removed
        for key in ['detections', 'audio', 'sr', 'tmp_audio_path', 'uploaded_filename', 'just_completed', 'previous_model', 'truncated_audio_path', 'original_duration', 'was_truncated', 'detection_in_progress', 'model_path', 'in_waiting_pool', 'concurrency_manager_acquired', 'queue_check_count']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Also remove from rate limiter waiting pool if user was waiting
        if CONCURRENCY_CONTROL_ENABLED and concurrency_manager and 'session_id' in st.session_state:
            try:
                # Remove from waiting pool (don't call finish_detection as user might not have been active)
                concurrency_manager.remove_from_waiting_pool(st.session_state['session_id'])
                # Also finish detection if they were active
                concurrency_manager.finish_detection(st.session_state['session_id'])
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Check if a new file was uploaded and clear previous results
    if uploaded_file is not None:
        current_filename = uploaded_file.name
        if 'uploaded_filename' in st.session_state and st.session_state['uploaded_filename'] != current_filename:
            # Clean up previous truncated file if it exists
            truncated_path = st.session_state.get('truncated_audio_path')
            if truncated_path is not None and os.path.exists(truncated_path):
                try:
                    os.unlink(truncated_path)
                except Exception:
                    pass  # Ignore errors during cleanup
            
            # Clear all detection results and queue state when a new file is uploaded
            for key in ['detections', 'audio', 'sr', 'tmp_audio_path', 'uploaded_filename', 'just_completed', 'previous_model', 'truncated_audio_path', 'original_duration', 'was_truncated', 'detection_in_progress', 'model_path', 'in_waiting_pool', 'concurrency_manager_acquired', 'queue_check_count']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Also remove from rate limiter queue if user was in queue
            if CONCURRENCY_CONTROL_ENABLED and concurrency_manager and 'session_id' in st.session_state:
                try:
                    # Remove from queue (don't call finish_detection as user might not have been active)
                    concurrency_manager.remove_from_waiting_pool(st.session_state['session_id'])
                    # Also finish detection if they were active
                    concurrency_manager.finish_detection(st.session_state['session_id'])
                except Exception:
                    pass  # Ignore errors during cleanup
        
        # Store current filename
        st.session_state['uploaded_filename'] = current_filename
        
        # Check audio duration and truncate if necessary
        # Skip if detections already exist (file already processed)
        if 'detections' not in st.session_state:
            MAX_DURATION_SECONDS = config.MAX_DURATION_SECONDS
            MAX_DURATION_MINUTES = MAX_DURATION_SECONDS / 60
            
            # Check if we need to process this file (new upload or not yet processed)
            # Process if we haven't checked duration yet, or if truncated file was deleted
            truncated_path = st.session_state.get('truncated_audio_path')
            if 'original_duration' not in st.session_state or (truncated_path is not None and not os.path.exists(truncated_path)):
                try:
                    # Load audio from uploaded file to check duration
                    # Use BytesIO to read from uploaded file directly
                    audio_bytes = io.BytesIO(uploaded_file.getvalue())
                    audio_check, sr_check = librosa.load(audio_bytes, sr=None)
                    original_duration = len(audio_check) / sr_check
                    
                    # Store original duration
                    st.session_state['original_duration'] = original_duration
                    
                    # Truncate if longer than MAX_DURATION_SECONDS
                    if original_duration > MAX_DURATION_SECONDS:
                        st.session_state['was_truncated'] = True
                        
                        # Reload and truncate audio to first MAX_DURATION_SECONDS
                        audio_bytes.seek(0)  # Reset to beginning
                        audio_truncated, sr_truncated = librosa.load(
                            audio_bytes,
                            sr=None,
                            duration=MAX_DURATION_SECONDS
                        )
                        
                        # Save truncated audio to a temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_truncated_file:
                            tmp_truncated_path = tmp_truncated_file.name
                        
                        # Save truncated audio using soundfile (more reliable than librosa.output)
                        sf.write(tmp_truncated_path, audio_truncated, sr_truncated)
                        
                        st.session_state['truncated_audio_path'] = tmp_truncated_path
                    else:
                        st.session_state['was_truncated'] = False
                        st.session_state['truncated_audio_path'] = None
                        
                except Exception as e:
                    # If truncation check fails, continue with original file
                    st.warning(f"⚠️ Could not check audio duration: {e}. Proceeding with original file.")
                    st.session_state['was_truncated'] = False
                    st.session_state['truncated_audio_path'] = None
                    st.session_state['original_duration'] = None
    
    # Lossy format warning
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext in ['.mp3', '.ogg']:
            st.warning("⚠️ Lossy format detected. Use WAV/FLAC for best results.")
    
    # Persistent truncation warning (show if file was truncated, but hide during and after detection)
    if (uploaded_file is not None and 
        st.session_state.get('was_truncated', False) and 
        not st.session_state.get('detection_in_progress', False) and
        'detections' not in st.session_state):
        MAX_DURATION_SECONDS = config.MAX_DURATION_SECONDS
        MAX_DURATION_MINUTES = MAX_DURATION_SECONDS / 60
        original_duration = st.session_state.get('original_duration', 0)
        if original_duration > 0:
            st.warning(
                f"⚠️ **Audio file truncated:** The uploaded file is about {original_duration/60:.0f} minutes long. "
                f"It has been automatically truncated to the first {MAX_DURATION_MINUTES:.0f} minutes for processing. "
                f"Only the first {MAX_DURATION_MINUTES:.0f} minutes will be analyzed."
            )
    
    # Check rate limiting status ONLY when actually needed (not on every rerun)
    # Only check if detection is in progress or user has explicitly requested status
    concurrency_status = None
    if st.session_state.get('detection_in_progress', False) and CONCURRENCY_CONTROL_ENABLED and concurrency_manager:
        # Only check status when detection is in progress and rate limiting is enabled
        concurrency_status = concurrency_manager.get_status(session_id)
    else:
        # No active detection or rate limiting disabled - use default empty status
        max_concurrent = concurrency_config.max_concurrent_detections if concurrency_config else 0
        concurrency_status = {
            'is_active': False,
            'is_waiting': False,
            'active_detections': 0,
            'waiting_pool_size': 0,
            'max_concurrent': max_concurrent,
            'can_make_request': True
        }
    
    # Process button (hide if results already exist, detection in progress, or user is in waiting pool)
    if uploaded_file is not None and 'detections' not in st.session_state and not st.session_state.get('detection_in_progress', False) and not st.session_state.get('in_waiting_pool', False):
        # Show the button - rate limiting check happens when button is clicked
        # Don't check status until button is clicked to avoid infinite loops
        if st.button("Detect Bird Calls", type="primary"):
            if CONCURRENCY_CONTROL_ENABLED and concurrency_manager:
                # Now check if we can actually start (this may add to waiting pool)
                can_start, reason = concurrency_manager.can_start_detection(session_id)
                
                if can_start:
                    # Set flag immediately to hide button during detection
                    st.session_state['detection_in_progress'] = True
                    st.rerun()
                else:
                    # Check if user was added to waiting pool
                    # Use get_status() instead of is_in_waiting_pool() for compatibility
                    status = concurrency_manager.get_status(session_id)
                    is_waiting = status.get('is_waiting', False)
                    was_already_waiting = st.session_state.get('in_waiting_pool', False)
                    
                    if is_waiting:
                        st.session_state['in_waiting_pool'] = True
                        # Rerun immediately so the persistent waiting message section handles the display
                        # This prevents showing a duplicate warning from the button handler
                        st.rerun()
                    else:
                        # Not in waiting pool - show the reason (e.g., rate limit)
                        st.warning(f"⚠️ {reason}")
                        # Don't rerun - let user see the message
            else:
                # Rate limiting disabled - start immediately
                st.session_state['detection_in_progress'] = True
                st.rerun()
    
    # Show waiting pool status if user is waiting
    if st.session_state.get('in_waiting_pool') and not st.session_state.get('detection_in_progress', False):
        # Check current status - user might not be waiting anymore
        if CONCURRENCY_CONTROL_ENABLED and concurrency_manager:
            current_status = concurrency_manager.get_status(session_id)
            is_waiting = current_status.get('is_waiting', False)
            
            if is_waiting:
                # Show waiting message
                st.warning("⚠️ Server is busy. Try again later.")
            else:
                # No longer waiting - clear the state
                del st.session_state['in_waiting_pool']
                st.info("✅ No longer waiting. You can try again.")
                st.rerun()
                return
        
        if st.button("Refresh Status & Try to Start", key="refresh_waiting_status"):
            # Try to start detection directly
            if CONCURRENCY_CONTROL_ENABLED and concurrency_manager:
                if concurrency_manager.start_detection(session_id):
                    # Successfully acquired!
                    st.session_state['detection_in_progress'] = True
                    st.session_state['concurrency_manager_acquired'] = True
                    del st.session_state['in_waiting_pool']
                    st.rerun()
                else:
                    # Still can't acquire - check status
                    fresh_status = concurrency_manager.get_status(session_id)
                    if not fresh_status.get('is_waiting', False):
                        # Not in waiting pool anymore - clear state
                        del st.session_state['in_waiting_pool']
                    st.rerun()
    
    # Show detection progress if in progress
    if st.session_state.get('detection_in_progress', False) and uploaded_file is not None:
        
        # Check if we've already acquired the rate limiter slot
        if CONCURRENCY_CONTROL_ENABLED and concurrency_manager and 'concurrency_manager_acquired' not in st.session_state:
            # Show status message while acquiring
            status_placeholder = st.empty()
            status_placeholder.info("🔄 Acquiring processing slot...")
            
            # Get fresh status
            concurrency_status = concurrency_manager.get_status(session_id)
            is_waiting = concurrency_status.get('is_waiting', False)
            is_active = concurrency_status.get('is_active', False)
            
            if is_active:
                # Already active - mark as acquired and proceed
                status_placeholder.empty()
                st.session_state['concurrency_manager_acquired'] = True
            elif is_waiting:
                # In waiting pool - show status message
                status_placeholder.info("⏳ Waiting for a processing slot... Processing will start automatically when a slot becomes available.")
                
                # Try to acquire (non-blocking check)
                # Just check status to see if we're active now
                fresh_status = concurrency_manager.get_status(session_id)
                if fresh_status.get('is_active', False):
                    # We're now active - mark as acquired
                    status_placeholder.empty()
                    st.session_state['concurrency_manager_acquired'] = True
                    # Clear waiting pool state
                    if 'in_waiting_pool' in st.session_state:
                        del st.session_state['in_waiting_pool']
                elif not fresh_status.get('is_waiting', False):
                    # Not in waiting pool anymore - try to start directly
                    if concurrency_manager.start_detection(session_id):
                        status_placeholder.empty()
                        st.session_state['concurrency_manager_acquired'] = True
                        # Clear waiting pool state
                        if 'in_waiting_pool' in st.session_state:
                            del st.session_state['in_waiting_pool']
                    else:
                        # Still can't acquire - might have been added back to pool
                        fresh_status2 = concurrency_manager.get_status(session_id)
                        if fresh_status2.get('is_waiting', False):
                            # Still waiting
                            status_placeholder.info("⏳ Still waiting for a processing slot... Processing will start automatically when a slot becomes available.")
                        else:
                            status_placeholder.warning("⚠️ Failed to acquire slot. Please try again.")
                            del st.session_state['detection_in_progress']
                            if 'in_waiting_pool' in st.session_state:
                                del st.session_state['in_waiting_pool']
                            return
                
                # Show button to manually check and try to acquire
                if st.button("Check Status & Try to Start", key="check_waiting_status"):
                    # Try to start detection directly - this will remove from pool and acquire if slot available
                    if concurrency_manager.start_detection(session_id):
                        # Successfully acquired!
                        st.session_state['concurrency_manager_acquired'] = True
                        if 'in_waiting_pool' in st.session_state:
                            del st.session_state['in_waiting_pool']
                        st.rerun()
                    else:
                        # Still can't acquire - check status
                        fresh_status = concurrency_manager.get_status(session_id)
                        if fresh_status.get('is_waiting', False):
                            st.info("⏳ Still waiting for a processing slot.")
                        elif fresh_status.get('is_active'):
                            # Somehow active now
                            st.session_state['concurrency_manager_acquired'] = True
                            if 'in_waiting_pool' in st.session_state:
                                del st.session_state['in_waiting_pool']
                            st.rerun()
                        else:
                            st.warning("⚠️ Slot not available yet. Please wait.")
                        st.rerun()
                return
            else:
                # Not in queue and not active - try to acquire
                status_placeholder.info("🔄 Checking availability and acquiring slot...")
                
                can_start, reason = concurrency_manager.can_start_detection(session_id)
                if not can_start:
                    # Can't start - show reason and clear flag
                    status_placeholder.warning(f"⚠️ {reason}")
                    del st.session_state['detection_in_progress']
                    return
                
                # Can start - acquire the slot (non-blocking with timeout)
                status_placeholder.info("🔄 Acquiring processing slot...")
                
                # Try to acquire - this should be immediate since can_start_detection said we can start
                try:
                    acquired = concurrency_manager.start_detection(session_id)
                    if acquired:
                        status_placeholder.empty()
                        st.session_state['concurrency_manager_acquired'] = True
                    else:
                        # Failed to acquire - might have been added to waiting pool
                        # Check status again
                        fresh_status = concurrency_manager.get_status(session_id)
                        if fresh_status.get('is_waiting', False):
                            # Show waiting message
                            status_placeholder.info("⏳ Waiting for a processing slot... Processing will start automatically when a slot becomes available.")
                            st.session_state['in_waiting_pool'] = True
                            if st.button("Check Status", key="check_waiting_after_fail"):
                                st.rerun()
                            return
                        else:
                            status_placeholder.error("⚠️ Failed to acquire processing slot. The slot may have been taken by another user. Please try again.")
                            del st.session_state['detection_in_progress']
                            return
                except Exception as e:
                    # Error during acquisition
                    status_placeholder.error(f"⚠️ Error acquiring slot: {e}")
                    del st.session_state['detection_in_progress']
                    import traceback
                    st.code(traceback.format_exc())
                    return
        elif not CONCURRENCY_CONTROL_ENABLED or not concurrency_manager:
            # Rate limiting disabled - mark as acquired and proceed
            st.session_state['concurrency_manager_acquired'] = True
        
        # Now process the detection
        with st.spinner("Processing audio file..."):
            try:
                # Use truncated file if available, otherwise save uploaded file to temporary location
                truncated_path = st.session_state.get('truncated_audio_path')
                if truncated_path is not None and os.path.exists(truncated_path):
                    # Use the pre-truncated file
                    tmp_audio_path = truncated_path
                else:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_audio_path = tmp_file.name
                
                # Initialize detector with species mapping
                # Note: Each user session gets its own detector instance to avoid thread-safety issues
                detector = BirdCallDetector(
                    model_path=selected_model,
                    species_mapping=st.session_state['species_mapping'],
                    conf_threshold=conf_threshold,
                    nms_iou_threshold=NMS_IOU_THRESHOLD,
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
                # Note: We don't store the detector object itself to avoid memory issues and potential
                # thread-safety problems. The detector is recreated if needed for spectrogram generation.
                st.session_state['detections'] = detections
                st.session_state['audio'] = audio
                st.session_state['sr'] = sr
                # Store model path and settings instead of detector object
                st.session_state['model_path'] = selected_model
                st.session_state['tmp_audio_path'] = tmp_audio_path
                st.session_state['just_completed'] = True
                
                # Clear detection in progress flag and rate limiter acquired flag
                del st.session_state['detection_in_progress']
                if 'concurrency_manager_acquired' in st.session_state:
                    del st.session_state['concurrency_manager_acquired']
                
                # Release rate limiter slot
                if CONCURRENCY_CONTROL_ENABLED and concurrency_manager:
                    concurrency_manager.finish_detection(session_id)
                
                # Rerun to show results
                st.rerun()
                
            except Exception as e:
                # Clear flags on error
                if 'detection_in_progress' in st.session_state:
                    del st.session_state['detection_in_progress']
                if 'concurrency_manager_acquired' in st.session_state:
                    del st.session_state['concurrency_manager_acquired']
                # Release rate limiter slot on error
                concurrency_manager.finish_detection(session_id)
                st.error(f"Error processing audio: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results if available
    if 'detections' in st.session_state:
        detections = st.session_state['detections']
        audio = st.session_state['audio']
        sr = st.session_state['sr']
        # Detector is not stored in session state to avoid memory issues and thread-safety problems
        
        # Show success message if just completed (will disappear after spectrogram renders)
        show_success_message = st.session_state.get('just_completed', False)
        success_placeholder = st.empty()
        
        if show_success_message:
            success_placeholder.info(f"Detection complete! Found {len(detections)} bird call segments.")
            st.session_state['just_completed'] = False
        
        st.markdown("---")
        # st.header("Detection Results")
        
        # PCEN Spectrogram with Detections (shown first)
        st.subheader("PCEN Spectrogram with Detections")
        
        duration = len(audio) / sr
        duration_info = f"**Audio duration:** {duration:.1f}s"
        if st.session_state.get('was_truncated', False):
            original_duration = st.session_state.get('original_duration', duration)
            duration_info += f" (truncated from {original_duration/60:.1f} min)"
        st.write(f"{duration_info} | **Detections:** {len(detections)} | Scroll to navigate through the audio timeline")
        
        # Generate spectrogram with species-specific colors
        species_mappings = st.session_state.get('species_mappings', {})
        if not species_mappings or 'bird_colors' not in species_mappings:
            # Fallback to default species mapping if mappings not available
            default_config = config.get_species_mapping('Hawaii')
            bird_colors = default_config['bird_colors']
        else:
            bird_colors = species_mappings['bird_colors']
        
        with st.spinner("Generating spectrogram with PCEN and adding bounding boxes..."):
            try:
                full_spectrogram = create_full_spectrogram_visualization(audio, sr, detections, bird_colors=bird_colors)
                # Validate the image was created successfully
                if full_spectrogram is None:
                    raise RuntimeError("Spectrogram generation returned None")
                if full_spectrogram.size[0] == 0 or full_spectrogram.size[1] == 0:
                    raise RuntimeError(f"Invalid spectrogram dimensions: {full_spectrogram.size}")
            except Exception as e:
                st.error(f"⚠️ Error generating spectrogram: {e}")
                import traceback
                st.code(traceback.format_exc())
                # Create a placeholder image with error message
                duration = len(audio) / sr
                width_pixels = max(100, int(duration * 100))
                full_spectrogram = Image.new('RGB', (width_pixels, 600), color='black')
                # Add error text to the image
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(full_spectrogram)
                try:
                    # Try to use a default font
                    font = ImageFont.load_default()
                except:
                    font = None
                error_text = f"Error: {str(e)[:50]}"
                draw.text((10, 10), error_text, fill='red', font=font)
        
        # Display spectrogram in scrollable container
        if full_spectrogram:
            with st.spinner("Rendering spectrogram..."):
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
        
        # Wait 3 seconds after spectrogram is rendered, then remove the success message
        if show_success_message:
            time.sleep(3)  # Keep message visible for 3 seconds
            success_placeholder.empty()  # Remove completely to avoid blank space
        
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
                avg_conf = sum(d['avg_confidence'] if 'detections_merged' in d else d['confidence'] for d in detections) / len(detections)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        with col4:
            if detections:
                total_duration = sum(d['time_end'] - d['time_start'] for d in detections)
                st.metric("Total Duration", f"{total_duration:.1f}s")
        
        # Species breakdown
        if detections:
            st.subheader("Species Breakdown")
            # Get species mappings for common name lookup
            species_mappings = st.session_state.get('species_mappings', {})
            
            species_counts = {}
            for det in detections:
                species = det['species']
                if species not in species_counts:
                    species_counts[species] = 0
                species_counts[species] += 1
            
            # Convert eBird codes to common names for display
            species_list = []
            for ebird_code, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
                common_name = get_common_name_from_ebird_code(ebird_code, species_mappings)
                species_list.append({'Species': common_name, 'Count': count})
            
            species_df = pd.DataFrame(species_list)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(species_df, width='stretch', hide_index=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                try:
                    species_df_plot = species_df.head(10)  # Top 10 species
                    ax.barh(species_df_plot['Species'], species_df_plot['Count'])
                    ax.set_xlabel('Count')
                    ax.set_title('Top Species Detected')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                finally:
                    # Always close figure to prevent memory leaks and conflicts in multi-user scenarios
                    plt.close(fig)
                    plt.clf()
        
        # Detection table
        st.markdown("---")
        st.subheader("Detailed Detection Table")
        
        # Get species mappings for common name lookup
        species_mappings = st.session_state.get('species_mappings', {})
        df = format_detections_for_table(detections, species_mappings)
        if not df.empty:
            st.dataframe(df, width='stretch', hide_index=True, height=400)
        
        # Download section
        st.markdown("---")
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            json_data = {
                'audio_file': uploaded_file.name if 'uploaded_file' in locals() else 'unknown',
                'model_config': {
                    'model': str(selected_model),
                    'confidence_threshold': conf_threshold,
                    'nms_iou_threshold': NMS_IOU_THRESHOLD,
                    'song_gap_threshold': song_gap_threshold,
                    'species_mapping': st.session_state.get('species_mapping', 'Hawaii'),  # Fallback to default species mapping
                },
                'detection_count': len(detections),
                'detections': detections
            }
            
            # Convert numpy types to JSON-serializable types
            json_data = convert_to_json_serializable(json_data)
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="Download JSON",
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
                    'eBird Code': det['species'],
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_str = csv_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
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

