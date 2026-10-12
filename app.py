import base64
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# SECTION 1: CORE IMAGE PROCESSING ALGORITHMS (IVP Course Algorithms)
# Classical spatial and point operations implemented with OpenCV & NumPy.
# ==============================================================================

def convert_to_grayscale(image):
    """
    Converts a 3-channel RGB image to single-channel Grayscale using
    the standard ITU-R BT.601 luminance weighting:
    Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def convert_to_binary(image, threshold_value=128):
    """
    Converts an image to binary black-and-white using global thresholding:
    dst(x,y) = 255 if src(x,y) > threshold else 0
    """
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def apply_gaussian_blur(image, kernel_size=1):
    """
    Applies Gaussian low-pass spatial filtering for noise reduction and smoothing.
    Kernel size must be positive and odd.
    """
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_sharpening_kernel(image, k):
    """
    Enhances high spatial frequencies using a 2D Laplacian-derived convolution kernel.
    Clips output to [0, 255] to prevent arithmetic wrap-around.
    Kernel matrix:
        [   0,         -k,          0      ]
        [  -k,     1 + 4*k,        -k      ]
        [   0,         -k,          0      ]
    """
    if k <= 0.0:
        return image
    kernel = np.array([
        [0.0, -k, 0.0],
        [-k, 1.0 + 4.0 * k, -k],
        [0.0, -k, 0.0]
    ], dtype=np.float32)
    filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def adjust_contrast(image, gain):
    """
    Applies centered linear contrast scaling around midpoint 128:
    g(x,y) = clip((f(x,y) - 128) * gain + 128, 0, 255)
    """
    transformed = (image.astype(np.float32) - 128.0) * gain + 128.0
    return np.clip(transformed, 0, 255).astype(np.uint8)


def adjust_colour(image, sat_gain):
    """
    Scales chromatic saturation in HSV color space.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_warmth(image, warmth_delta):
    """
    Applies spectral warmth bias (boosts Red, gently cools Blue).
    """
    if len(image.shape) != 3 or image.shape[2] != 3 or warmth_delta == 0:
        return image
    res = image.astype(np.float32)
    res[:, :, 0] = np.clip(res[:, :, 0] + warmth_delta, 0, 255)
    res[:, :, 2] = np.clip(res[:, :, 2] - warmth_delta * 0.7, 0, 255)
    return np.clip(res, 0, 255).astype(np.uint8)


def apply_grain(image, sigma):
    """
    Synthesizes zero-mean Gaussian film noise.
    """
    if sigma <= 0.0:
        return image
    noise = np.random.normal(0, sigma, image.shape)
    res = image.astype(np.float32) + noise
    return np.clip(res, 0, 255).astype(np.uint8)


def apply_histogram_equalization(image):
    """
    Enhances global image contrast via histogram equalization.
    For color images: converts to YCrCb space, equalizes luminance only,
    and returns to RGB to preserve color hues without chrominance artifacts.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return image


def apply_master_tuning(image, sharpen_val, contrast_val, colour_val, warmth_val, grain_val):
    """
    Executes the real-time IVP studio processing pipeline corresponding
    to the 5 live sliders in the reference design.
    """
    res = image.copy()
    
    # 1. Sharpening: Laplacian spatial kernel
    k = round((sharpen_val / 100.0) * 1.3, 2)
    if k > 0:
        res = apply_sharpening_kernel(res, k)
        
    # 2. Linear Contrast Stretch
    c_gain = round(1.0 + (contrast_val - 50) * 0.0267, 2)
    if c_gain != 1.0:
        res = adjust_contrast(res, c_gain)
        
    # 3. Colour (Saturation in HSV)
    sat_gain = round(1.0 + (colour_val - 50) * 0.02, 2)
    if sat_gain != 1.0 and len(res.shape) == 3:
        res = adjust_colour(res, sat_gain)
        
    # 4. Spectral Warmth
    w_delta = (warmth_val - 50) * 0.8
    if w_delta != 0 and len(res.shape) == 3:
        res = adjust_warmth(res, w_delta)
        
    # 5. Film Grain (Gaussian perturbations)
    sigma = (grain_val / 100.0) * 15.0
    if sigma > 0:
        res = apply_grain(res, sigma)
        
    return res


def img_to_base64_jpeg(image_np, quality=88):
    """Converts a NumPy RGB image to Base64 JPEG string for fast browser rendering."""
    pil_img = Image.fromarray(image_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_sample_image():
    """Loads the built-in comic superhero sample image or creates fallback."""
    sample_path = Path(__file__).parent / "sample.jpg"
    if sample_path.exists():
        try:
            raw = Image.open(sample_path).convert("RGB")
            return np.array(raw)
        except Exception:
            pass
    
    # Procedural geometric fallback if sample.jpg not present
    fallback = np.zeros((450, 750, 3), dtype=np.uint8)
    for y in range(450):
        for x in range(750):
            fallback[y, x] = [int(15 + 40 * (x / 750)), int(25 + 60 * (y / 450)), int(70 + 80 * (x / 750))]
    cv2.circle(fallback, (375, 225), 110, (255, 95, 45), -1)
    cv2.putText(fallback, "REVEAL STUDIO", (230, 235), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
    return fallback


# ==============================================================================
# SECTION 2: HIGH-FIDELITY CSS ARCHITECTURE (REVEAL STUDIO / NETRAYA DESIGN)
# Midnight navy, cyan halftone dots, warm coral glows, glass panels, sci-fi accents
# ==============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Playfair+Display:ital,wght@1,600;1,700;1,800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-midnight: #070B18;
    --navy-surface: #0E1626;
    --navy-card: rgba(14, 22, 38, 0.85);
    --accent-orange: #FF6B35;
    --accent-orange-bright: #FF3815;
    --accent-yellow: #FFB703;
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-glass-strong: rgba(255, 255, 255, 0.16);
    --text-primary: #F8FAFC;
    --text-muted: #8A9BB8;
    --text-slate: #64748B;
}

/* Base Viewport & Halftone Dot Background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-midnight) !important;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(45, 95, 230, 0.35) 0%, transparent 45%),
        radial-gradient(circle at 90% 75%, rgba(255, 75, 43, 0.15) 0%, transparent 45%),
        radial-gradient(rgba(50, 115, 245, 0.24) 1.5px, transparent 1.5px) !important;
    background-size: 100% 100%, 100% 100%, 20px 20px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Hide Default Streamlit Chrome */
header[data-testid="stHeader"] {
    display: none !important;
}
#MainMenu, footer, [data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}

/* Main Container Layout */
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1360px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* -------------------------------------------------------------------------- */
/* 1. TOP NAVIGATION BAR                                                      */
/* -------------------------------------------------------------------------- */
.top-nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 28px;
    background: rgba(11, 17, 34, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid var(--border-glass);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 24px;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-target-icon {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    border: 1.5px solid #38B6FF;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: 0 0 10px rgba(56, 182, 255, 0.3);
}

.brand-target-cross-h {
    position: absolute;
    width: 22px;
    height: 1px;
    background: rgba(56, 182, 255, 0.7);
}

.brand-target-cross-v {
    position: absolute;
    height: 22px;
    width: 1px;
    background: rgba(56, 182, 255, 0.7);
}

.brand-target-dot {
    width: 6px;
    height: 6px;
    background: #FF3815;
    border-radius: 50%;
    z-index: 2;
    box-shadow: 0 0 8px #FF3815;
}

.brand-title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #FFFFFF;
}

.brand-tag-course {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: var(--accent-orange);
    letter-spacing: 0.12em;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(255, 107, 53, 0.12);
    border: 1px solid rgba(255, 107, 53, 0.25);
    margin-left: 6px;
}

.nav-links-row {
    display: flex;
    align-items: center;
    gap: 32px;
}

.nav-link-item {
    font-size: 13.5px;
    font-weight: 500;
    color: var(--text-muted);
    text-decoration: none !important;
    transition: color 0.2s ease;
}

.nav-link-item:hover {
    color: #FFFFFF;
}

.nav-right-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}

.btn-open-studio {
    background: linear-gradient(135deg, #FF6B35 0%, #FF3815 100%);
    color: #FFFFFF !important;
    font-size: 13px;
    font-weight: 700;
    padding: 8px 20px;
    border-radius: 24px;
    text-decoration: none !important;
    box-shadow: 0 4px 16px rgba(255, 80, 30, 0.4);
    transition: all 0.2s ease;
    display: inline-block;
}

.btn-open-studio:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(255, 80, 30, 0.6);
}

/* -------------------------------------------------------------------------- */
/* 2. HERO SECTION — LEFT COLUMN                                              */
/* -------------------------------------------------------------------------- */
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #7B8FA9;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.eyebrow-red-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #FF3815;
    box-shadow: 0 0 8px #FF3815;
}

.hero-main-title {
    font-size: 52px;
    font-weight: 800;
    line-height: 1.08;
    color: #FFFFFF;
    letter-spacing: -0.025em;
    margin-bottom: 18px;
}

.reveal-script {
    font-family: 'Playfair Display', 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-weight: 700;
    color: #FF6B35;
    position: relative;
    display: inline-block;
}

.reveal-underline-svg {
    position: absolute;
    bottom: -10px;
    left: 0;
    width: 100%;
    height: 12px;
}

.hero-supporting-text {
    font-size: 15px;
    line-height: 1.6;
    color: #94A3B8;
    max-width: 460px;
    margin-bottom: 24px;
}

.hero-cta-group {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 30px;
}

.btn-start-enhancing {
    background: linear-gradient(135deg, #FF6B35 0%, #FF3815 100%);
    color: #FFFFFF !important;
    font-size: 14px;
    font-weight: 700;
    padding: 12px 26px;
    border-radius: 30px;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 8px 24px rgba(255, 75, 43, 0.45);
    transition: all 0.2s ease;
}

.btn-start-enhancing:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(255, 75, 43, 0.6);
}

.btn-explore-features {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: #FFFFFF !important;
    font-size: 14px;
    font-weight: 600;
    padding: 12px 22px;
    border-radius: 30px;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s ease;
}

.btn-explore-features:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.4);
}

/* Feature Cards 3-in-a-row */
.feature-cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.feature-pill-card {
    background: rgba(16, 24, 42, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.feature-pill-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 107, 53, 0.35);
}

.feat-icon-badge {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: rgba(255, 56, 21, 0.18);
    border: 1px solid rgba(255, 56, 21, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FF6B35;
    font-size: 13px;
    flex-shrink: 0;
}

.feat-text-box {
    display: flex;
    flex-direction: column;
}

.feat-title {
    font-size: 12px;
    font-weight: 700;
    color: #F8FAFC;
}

.feat-sub {
    font-size: 10px;
    color: #8A9BB8;
    white-space: nowrap;
}

/* -------------------------------------------------------------------------- */
/* 3. HERO SECTION — RIGHT COLUMN (STUDIO STAGE & CONTROLS)                   */
/* -------------------------------------------------------------------------- */
.lightning-decor-row {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: -10px;
    padding-right: 28px;
    position: relative;
    z-index: 15;
}
.bolt-yellow { color: #FFE600; font-size: 22px; filter: drop-shadow(0 0 6px #FFE600); transform: rotate(-10deg); }
.bolt-orange { color: #FF6B35; font-size: 26px; filter: drop-shadow(0 0 6px #FF6B35); transform: rotate(15deg); }
.bolt-blue { color: #38B6FF; font-size: 22px; filter: drop-shadow(0 0 6px #38B6FF); transform: rotate(-5deg); }

/* Control Panel Docked Under Image */
.control-dock-card {
    background: rgba(14, 21, 38, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 16px 20px 14px 20px;
    margin-top: 10px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
}

/* Custom Streamlit Sliders inside Dock */
[data-testid="stSlider"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: #8899B5 !important;
    text-transform: uppercase !important;
    margin-bottom: -4px !important;
}

[data-testid="stSlider"] div[role="slider"] {
    background: #FFFFFF !important;
    border: 2px solid #FF6B35 !important;
    box-shadow: 0 0 8px rgba(255, 107, 53, 0.6) !important;
    width: 14px !important;
    height: 14px !important;
}

[data-testid="stSlider"] div[data-testid="stSliderTickBar"] {
    display: none !important;
}

/* Live Technical Readouts */
.tech-readout-wrap {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px;
    color: #7E90AC;
    line-height: 1.55;
    padding-top: 4px;
}

.tech-row-highlight {
    color: #FFB703;
    font-weight: 600;
}

/* Button Customizations */
.btn-auto-wrap div.stButton > button {
    background: linear-gradient(135deg, #FF6B35 0%, #FF3815 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 11.5px !important;
    padding: 6px 16px !important;
    box-shadow: 0 4px 14px rgba(255, 75, 43, 0.45) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.btn-auto-wrap div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255, 75, 43, 0.65) !important;
}

.btn-reset-wrap div.stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 11.5px !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.btn-reset-wrap div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    color: #FFFFFF !important;
}

/* -------------------------------------------------------------------------- */
/* 4. STUDIO WORKSPACE & UPLOAD LAB SECTION                                   */
/* -------------------------------------------------------------------------- */
.workspace-section-card {
    background: rgba(14, 22, 38, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 18px;
    padding: 24px 28px;
    margin-top: 36px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45);
}

.ws-header-group {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    padding-bottom: 16px;
    margin-bottom: 20px;
}

.ws-title-kicker {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: var(--accent-orange);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.ws-title-main {
    font-size: 20px;
    font-weight: 800;
    color: #FFFFFF;
}

.active-file-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(8, 13, 26, 0.7);
    border: 1px solid var(--border-glass-strong);
    border-radius: 14px;
    padding: 12px 18px;
    margin-top: 12px;
}

.active-file-thumb {
    width: 48px;
    height: 48px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.active-file-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.active-file-name {
    font-size: 13px;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 8px;
}

.chip-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.08);
    color: #94A3B8;
}

.chip-tag-orange {
    background: rgba(255, 107, 53, 0.2);
    color: #FFB703;
    border: 1px solid rgba(255, 107, 53, 0.4);
}

/* Feature Explanations Cards */
.feat-explain-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 24px;
}

.feat-explain-card {
    background: rgba(11, 17, 34, 0.65);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.feat-explain-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 107, 53, 0.3);
}

.feat-explain-title {
    font-size: 13.5px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 6px;
}

.feat-explain-desc {
    font-size: 11.5px;
    color: #8A9BB8;
    line-height: 1.5;
}

/* Download Button Primary Styling */
.btn-download-wrap div.stDownloadButton > button {
    background: linear-gradient(135deg, #FF6B35 0%, #FF3815 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 13.5px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 20px rgba(255, 75, 43, 0.45) !important;
    width: 100% !important;
}

/* Footer */
.footer-wrap {
    text-align: center;
    padding: 40px 0 20px 0;
    color: #64748B;
    font-size: 12px;
}

.footer-brand {
    font-family: 'DM Mono', monospace;
    font-size: 11.5px;
    font-weight: 700;
    color: #CBD5E1;
    letter-spacing: 0.14em;
    margin-bottom: 6px;
}
</style>
"""

# ==============================================================================
# SECTION 3: APPLICATION CONTROLLER & STATE INITIALIZATION
# ==============================================================================

def init_session_state():
    """Initializes widget states cleanly to avoid check_session_state_rules conflicts."""
    defaults = {
        "sharpen": 68,
        "contrast": 62,
        "colour": 72,
        "warmth": 58,
        "grain": 22,
        "active_ivp_tab": "Master Studio Tuning",
        "uploader_key_id": 0,
        "use_custom_upload": False,
        "gray_blend": 100,
        "thresh_val": 128,
        "blur_val": 7,
        "hist_blend": 100,
        "uploaded_image_cache": None,
        "uploaded_filename_cache": None,
        "uploaded_filesize_cache": 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_auto_enhance():
    """Applies the reference screenshot auto-enhance preset."""
    st.session_state["sharpen"] = 68
    st.session_state["contrast"] = 62
    st.session_state["colour"] = 72
    st.session_state["warmth"] = 58
    st.session_state["grain"] = 22


def reset_to_baseline():
    """Resets sliders to neutral original image baseline."""
    st.session_state["sharpen"] = 0
    st.session_state["contrast"] = 50
    st.session_state["colour"] = 50
    st.session_state["warmth"] = 50
    st.session_state["grain"] = 0


def revert_to_sample():
    """Clears uploaded image cache and reverts to built-in sample."""
    st.session_state["uploaded_image_cache"] = None
    st.session_state["uploaded_filename_cache"] = None
    st.session_state["uploaded_filesize_cache"] = 0
    st.session_state["use_custom_upload"] = False
    st.session_state["uploader_key_id"] += 1


# ==============================================================================
# SECTION 4: MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    # 0. Page Configuration
    st.set_page_config(
        page_title="Reveal Studio // NETRAYA Image Studio",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    init_session_state()

    # Inject Matching Styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 1. TOP NAVIGATION BAR (Target Icon + Reveal Studio + Nav Links + Open Studio)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <header class="top-nav-bar" id="home">
            <div class="brand-wrapper">
                <div class="brand-target-icon">
                    <div class="brand-target-cross-h"></div>
                    <div class="brand-target-cross-v"></div>
                    <div class="brand-target-dot"></div>
                </div>
                <div class="brand-title">Reveal Studio</div>
                <span class="brand-tag-course">NETRAYA // IVP</span>
            </div>
            <nav class="nav-links-row">
                <a href="#features" class="nav-link-item">Features</a>
                <a href="#how-it-works" class="nav-link-item">How it works</a>
                <a href="#pricing" class="nav-link-item">Pricing</a>
            </nav>
            <div class="nav-right-actions">
                <a href="#editor-section" class="btn-open-studio">Open studio</a>
            </div>
        </header>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 2. IMAGE RETRIEVAL & PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    # Determine active source image: uploaded file OR high-quality sample image
    active_image = None
    active_filename = "sample_comic_superhero.jpg"
    is_custom_file = False
    file_bytes_len = 0

    if "uploaded_image_cache" in st.session_state and st.session_state["uploaded_image_cache"] is not None:
        active_image = st.session_state["uploaded_image_cache"]
        active_filename = st.session_state.get("uploaded_filename_cache", "uploaded_image.png")
        file_bytes_len = st.session_state.get("uploaded_filesize_cache", 0)
        is_custom_file = True
    else:
        active_image = load_sample_image()
        active_filename = "sample_hero_graphic.jpg"
        file_bytes_len = 1181175

    img_h, img_w = active_image.shape[:2]

    # Compute Processed Image using the 5 live sliders
    s_val = st.session_state["sharpen"]
    c_val = st.session_state["contrast"]
    col_val = st.session_state["colour"]
    w_val = st.session_state["warmth"]
    g_val = st.session_state["grain"]

    processed_image = apply_master_tuning(
        active_image,
        sharpen_val=s_val,
        contrast_val=c_val,
        colour_val=col_val,
        warmth_val=w_val,
        grain_val=g_val
    )

    # Generate Base64 representations for 60fps comparison stage
    b64_orig = img_to_base64_jpeg(active_image, quality=86)
    b64_enh = img_to_base64_jpeg(processed_image, quality=86)

    # --------------------------------------------------------------------------
    # 3. HERO SECTION — TWO COLUMN LAYOUT (Matching studio-desktop-combined.png)
    # --------------------------------------------------------------------------
    hero_col_left, hero_col_right = st.columns([1.0, 1.25], gap="large")

    # --- LEFT HERO COLUMN: Branding, Headline, Actions, 3 Feature Cards ---
    with hero_col_left:
        st.markdown(
            """
            <div class="hero-left-container">
                <div class="hero-eyebrow">
                    <span class="eyebrow-red-dot"></span>
                    <span>IMAGE &amp; VIDEO PROCESSING COURSEWARE</span>
                </div>
                <h1 class="hero-main-title">
                    Every image<br>
                    has more to<br>
                    <span class="reveal-script">
                        reveal.
                        <svg class="reveal-underline-svg" viewBox="0 0 160 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 9.5C45 2.5 115 2 157 10.5" stroke="#FF6B35" stroke-width="3.5" stroke-linecap="round"/>
                        </svg>
                    </span>
                </h1>
                <p class="hero-supporting-text">
                    A hands-on enhancer studio. Sharpen, rebalance and refine your frames with live controls, and see what each tool does under the hood.
                </p>
                <div class="hero-cta-group">
                    <a href="#editor-section" class="btn-start-enhancing">
                        Start enhancing &rarr;
                    </a>
                    <a href="#features" class="btn-explore-features">
                        &#9658; Explore features
                    </a>
                </div>
                <div class="feature-cards-grid">
                    <div class="feature-pill-card">
                        <div class="feat-icon-badge">&#10022;</div>
                        <div class="feat-text-box">
                            <span class="feat-title">AI powered</span>
                            <span class="feat-sub">Image enhancement</span>
                        </div>
                    </div>
                    <div class="feature-pill-card">
                        <div class="feat-icon-badge">&#128737;</div>
                        <div class="feat-text-box">
                            <span class="feat-title">Privacy first</span>
                            <span class="feat-sub">Your images stay local</span>
                        </div>
                    </div>
                    <div class="feature-pill-card">
                        <div class="feat-icon-badge">&#9733;</div>
                        <div class="feat-text-box">
                            <span class="feat-title">High quality</span>
                            <span class="feat-sub">Clearer. Sharper. Better.</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- RIGHT HERO COLUMN: Interactive Split Stage & Docked Control Panel ---
    with hero_col_right:
        # Decorative Lightning Bolts (Top right)
        st.markdown(
            """
            <div class="lightning-decor-row">
                <span class="bolt-yellow">&#9889;</span>
                <span class="bolt-orange">&#9889;</span>
                <span class="bolt-blue">&#9889;</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Interactive Comparison Stage with Draggable Divider (Embedded HTML5 Component)
        comparison_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
        body {{ background: transparent; overflow: hidden; display: flex; justify-content: center; align-items: center; }}

        .stage-wrapper {{
            position: relative;
            width: 100%;
            max-width: 680px;
            height: 380px;
            border-radius: 16px;
            overflow: hidden;
            background: #0B1124;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.65);
        }}

        .img-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }}

        .img-enhanced-wrap {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            clip-path: polygon(var(--split, 50%) 0, 100% 0, 100% 100%, var(--split, 50%) 100%);
            pointer-events: none;
        }}

        /* Sci-fi Corner Brackets */
        .bracket {{
            position: absolute;
            width: 18px;
            height: 18px;
            pointer-events: none;
            z-index: 10;
        }}
        .bracket-tl {{ top: 12px; left: 12px; border-top: 2px solid rgba(255, 255, 255, 0.9); border-left: 2px solid rgba(255, 255, 255, 0.9); }}
        .bracket-tr {{ top: 12px; right: 12px; border-top: 2px solid rgba(255, 255, 255, 0.9); border-right: 2px solid rgba(255, 255, 255, 0.9); }}
        .bracket-bl {{ bottom: 12px; left: 12px; border-bottom: 2px solid rgba(255, 255, 255, 0.9); border-left: 2px solid rgba(255, 255, 255, 0.9); }}
        .bracket-br {{ bottom: 12px; right: 12px; border-bottom: 2px solid rgba(255, 255, 255, 0.9); border-right: 2px solid rgba(255, 255, 255, 0.9); }}

        /* Badges */
        .badge-orig {{
            position: absolute;
            top: 14px;
            left: 36px;
            background: rgba(11, 17, 36, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 3px 10px;
            color: #94A3B8;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.12em;
            font-family: monospace;
            z-index: 10;
        }}

        .badge-enh {{
            position: absolute;
            top: 14px;
            right: 36px;
            background: rgba(255, 183, 3, 0.2);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 183, 3, 0.5);
            border-radius: 6px;
            padding: 3px 10px;
            color: #FFB703;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.12em;
            font-family: monospace;
            z-index: 10;
        }}

        /* Timecode */
        .timecode {{
            position: absolute;
            bottom: 12px;
            right: 34px;
            color: #94A3B8;
            font-size: 10.5px;
            font-family: monospace;
            letter-spacing: 0.1em;
            z-index: 10;
        }}

        /* Draggable Divider Line & Handle */
        .divider-line {{
            position: absolute;
            top: 0;
            bottom: 0;
            left: var(--split, 50%);
            width: 2px;
            background: #FFFFFF;
            transform: translateX(-50%);
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.5);
            pointer-events: none;
            z-index: 8;
        }}

        .divider-handle {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 28px;
            height: 28px;
            background: #FFFFFF;
            border-radius: 50%;
            box-shadow: 0 2px 14px rgba(0, 0, 0, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0E1622;
            font-size: 13px;
            font-weight: 800;
        }}

        /* Range Drag Overlay for instant 60fps interaction */
        .range-drag-input {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: ew-resize;
            z-index: 20;
            margin: 0;
            padding: 0;
        }}
        </style>
        </head>
        <body>
        <div class="stage-wrapper" id="stage" style="--split: 50%;">
            <img src="data:image/jpeg;base64,{b64_orig}" class="img-layer" alt="Original" />
            <div class="img-enhanced-wrap">
                <img src="data:image/jpeg;base64,{b64_enh}" class="img-layer" alt="Enhanced" />
            </div>
            <div class="bracket bracket-tl"></div>
            <div class="bracket bracket-tr"></div>
            <div class="bracket bracket-bl"></div>
            <div class="bracket bracket-br"></div>
            <div class="badge-orig">ORIGINAL</div>
            <div class="badge-enh">ENHANCED</div>
            <div class="timecode">00:00:01:12</div>
            <div class="divider-line" id="dividerLine">
                <div class="divider-handle">&#x2194;</div>
            </div>
            <input type="range" min="0" max="100" value="50" class="range-drag-input" id="rangeInput" oninput="updateSplit(this.value)" />
        </div>
        <script>
        function updateSplit(val) {{
            document.getElementById('stage').style.setProperty('--split', val + '%');
        }}
        </script>
        </body>
        </html>
        """

        components.html(comparison_html, height=395, scrolling=False)

        # Translucent Control Panel Docked Under the Stage
        st.markdown("<div class='control-dock-card'>", unsafe_allow_html=True)

        # Row of 5 Adjustment Sliders Matching the Screenshot
        sl_c1, sl_c2, sl_c3, sl_c4, sl_c5 = st.columns(5)
        with sl_c1:
            st.slider("SHARPEN", 0, 100, key="sharpen")
        with sl_c2:
            st.slider("CONTRAST", 0, 100, key="contrast")
        with sl_c3:
            st.slider("COLOUR", 0, 100, key="colour")
        with sl_c4:
            st.slider("WARMTH", 0, 100, key="warmth")
        with sl_c5:
            st.slider("GRAIN", 0, 100, key="grain")

        # Bottom Bar: Live Technical Readouts & Action Buttons
        bot_col_left, bot_col_right = st.columns([1.5, 1.0])

        with bot_col_left:
            # Mathematical technical values dynamically calculated from active sliders
            k_calc = round((st.session_state["sharpen"] / 100.0) * 1.3, 2)
            center_k_calc = round(1.0 + 4.0 * k_calc, 2)
            c_mult_calc = round(1.0 + (st.session_state["contrast"] - 50) * 0.0267, 2)
            sat_mult_calc = round(1.0 + (st.session_state["colour"] - 50) * 0.02, 2)

            st.markdown(
                f"""
                <div class="tech-readout-wrap">
                    <div>kernel [0 -{k_calc:.2f} 0 | -{k_calc:.2f} {center_k_calc:.2f} -{k_calc:.2f} | 0 -{k_calc:.2f} 0]</div>
                    <div>contrast <span class="tech-row-highlight">x{c_mult_calc:.2f}</span> &nbsp;&bull;&nbsp; colour <span class="tech-row-highlight">x{sat_mult_calc:.2f}</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with bot_col_right:
            btn_col1, btn_col2 = st.columns([1.2, 0.9])
            with btn_col1:
                st.markdown("<div class='btn-auto-wrap'>", unsafe_allow_html=True)
                st.button("+ Auto-enhance", key="btn_auto_enh", on_click=set_auto_enhance)
                st.markdown("</div>", unsafe_allow_html=True)
            with btn_col2:
                st.markdown("<div class='btn-reset-wrap'>", unsafe_allow_html=True)
                st.button("Reset", key="btn_reset", on_click=reset_to_baseline)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True) # close control-dock-card

    # --------------------------------------------------------------------------
    # 4. STUDIO WORKSPACE & UPLOAD LAB (Immediately Accessible Below Hero)
    # --------------------------------------------------------------------------
    st.markdown("<div id='editor-section'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="workspace-section-card">
            <div class="ws-header-group">
                <div>
                    <div class="ws-title-kicker">STUDIO WORKSPACE &bull; IMAGE LAB</div>
                    <div class="ws-title-main">Active Inspection &amp; File Management</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    up_col1, up_col2 = st.columns([1.1, 1.0], gap="large")

    with up_col1:
        st.markdown(
            """
            <div style="font-size: 13.5px; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">
                Upload an Image (JPG, JPEG, PNG, WEBP)
            </div>
            <div style="font-size: 11.5px; color: #8A9BB8; margin-bottom: 12px;">
                Upload your custom image to calibrate with the live comparison stage above, or keep the initial superhero sample.
            </div>
            """,
            unsafe_allow_html=True
        )

        uploader_key = f"studio_uploader_{st.session_state['uploader_key_id']}"
        uploaded_file = st.file_uploader(
            "Upload image file",
            type=["jpg", "jpeg", "png", "webp"],
            key=uploader_key,
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.session_state.get("uploaded_filename_cache") != uploaded_file.name:
                try:
                    raw_user_img = Image.open(uploaded_file).convert("RGB")
                    user_np = np.array(raw_user_img)
                    # Cache user image into session state
                    st.session_state["uploaded_image_cache"] = user_np
                    st.session_state["uploaded_filename_cache"] = uploaded_file.name
                    st.session_state["uploaded_filesize_cache"] = uploaded_file.size
                    st.session_state["use_custom_upload"] = True
                    st.rerun()
                except Exception:
                    st.error("Failed to parse uploaded image. Please try another JPG or PNG.")

    with up_col2:
        # File status & Active file details card
        size_fmt = f"{file_bytes_len / 1024:.1f} KB" if file_bytes_len < 1024*1024 else f"{file_bytes_len / (1024*1024):.2f} MB"
        fmt_pill = active_filename.split(".")[-1].upper() if "." in active_filename else "PNG"

        # Active File Preview Card
        st.markdown(
            f"""
            <div class="active-file-card">
                <img src="data:image/jpeg;base64,{b64_orig}" class="active-file-thumb" alt="Thumbnail" />
                <div class="active-file-meta">
                    <div class="active-file-name">
                        <span>{active_filename}</span>
                        <span style="color: #22C55E; font-size: 12px;">&#10004; Active</span>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <span class="chip-tag chip-tag-orange">{fmt_pill}</span>
                        <span class="chip-tag">{img_w} &times; {img_h} PX</span>
                        <span class="chip-tag">{size_fmt}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Download & Reset Image Buttons
        d_col1, d_col2 = st.columns([1.3, 1.0])
        with d_col1:
            # Download processed image as PNG
            result_pil = Image.fromarray(processed_image)
            out_buf = io.BytesIO()
            result_pil.save(out_buf, format="PNG")
            out_bytes = out_buf.getvalue()

            st.markdown("<div class='btn-download-wrap' style='margin-top: 14px;'>", unsafe_allow_html=True)
            st.download_button(
                label="⤓ Download Enhanced Image",
                data=out_bytes,
                file_name=f"enhanced_{active_filename.rsplit('.', 1)[0]}.png",
                mime="image/png",
                width="stretch"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with d_col2:
            if is_custom_file:
                st.markdown("<div style='margin-top: 14px;'>", unsafe_allow_html=True)
                st.button("🔄 Revert to Sample", key="btn_revert_sample", on_click=revert_to_sample)
                st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. CLASSICAL IVP COURSEWARE SYLLABUS LAB
    # --------------------------------------------------------------------------
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 14px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;'>Classical IVP Syllabus Tools</div>", unsafe_allow_html=True)
    
    ivp_tab1, ivp_tab2, ivp_tab3, ivp_tab4 = st.tabs([
        "Grayscale Conversion (BT.601)",
        "Binary Thresholding (Otsu & Fixed)",
        "Gaussian Smoothing",
        "Histogram Equalization (YCrCb)"
    ])

    with ivp_tab1:
        c_g1, c_g2 = st.columns([1.2, 1.0])
        with c_g1:
            gray_res = convert_to_grayscale(active_image)
            st.image(gray_res, caption="Grayscale (ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B)", width="stretch")
        with c_g2:
            st.markdown(
                """
                <div style="font-size: 12.5px; color: #94A3B8; line-height: 1.6; padding: 10px;">
                    <b style="color: #FFF;">Luminance Transform:</b> Converts three chromatic channels into a single intensity channel weighting human eye sensitivity to green light over red and blue.
                </div>
                """,
                unsafe_allow_html=True
            )
            # Download grayscale option
            g_buf = io.BytesIO()
            Image.fromarray(gray_res).save(g_buf, format="PNG")
            st.download_button("⤓ Download Grayscale PNG", g_buf.getvalue(), file_name="grayscale_ivp.png", mime="image/png")

    with ivp_tab2:
        c_t1, c_t2 = st.columns([1.2, 1.0])
        with c_t2:
            t_val = st.slider("Threshold Value", 0, 255, 128, key="tab_thresh_slider")
            st.markdown(
                f"""
                <div style="font-size: 12.5px; color: #94A3B8; line-height: 1.6; padding: 10px;">
                    <b style="color: #FFF;">Global Point Segmentation:</b> Pixels with intensity &gt; {t_val} mapped to 255 (white); otherwise 0 (black).
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_t1:
            bin_res = convert_to_binary(active_image, threshold_value=t_val)
            st.image(bin_res, caption=f"Binary Thresholded (T = {t_val})", width="stretch")

    with ivp_tab3:
        c_b1, c_b2 = st.columns([1.2, 1.0])
        with c_b2:
            k_blur = st.slider("Gaussian Kernel Size", 1, 31, 7, step=2, key="tab_blur_slider")
            st.markdown(
                f"""
                <div style="font-size: 12.5px; color: #94A3B8; line-height: 1.6; padding: 10px;">
                    <b style="color: #FFF;">Spatial Low-Pass Smoothing:</b> Convolves image with a {k_blur}&times;{k_blur} 2D Gaussian bell-curve kernel to attenuate high-frequency noise.
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_b1:
            blur_res = apply_gaussian_blur(active_image, kernel_size=k_blur)
            st.image(blur_res, caption=f"Gaussian Blur (Kernel = {k_blur}×{k_blur})", width="stretch")

    with ivp_tab4:
        c_h1, c_h2 = st.columns([1.2, 1.0])
        with c_h1:
            eq_res = apply_histogram_equalization(active_image)
            st.image(eq_res, caption="Histogram Equalized (YCrCb Chrominance Preserving)", width="stretch")
        with c_h2:
            st.markdown(
                """
                <div style="font-size: 12.5px; color: #94A3B8; line-height: 1.6; padding: 10px;">
                    <b style="color: #FFF;">Luminance Contrast Equalization:</b> Equalizes the cumulative distribution function (CDF) of the Y luminance channel only in YCrCb space, avoiding unnatural color hue shifts.
                </div>
                """,
                unsafe_allow_html=True
            )
            h_buf = io.BytesIO()
            Image.fromarray(eq_res).save(h_buf, format="PNG")
            st.download_button("⤓ Download Equalized PNG", h_buf.getvalue(), file_name="equalized_ivp.png", mime="image/png")

    st.markdown("</div>", unsafe_allow_html=True) # close workspace-section-card

    # --------------------------------------------------------------------------
    # 6. FEATURES BREAKDOWN (Nav Anchor: #features)
    # --------------------------------------------------------------------------
    st.markdown("<div id='features'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-top: 40px; padding: 10px 0;">
            <div class="ws-title-kicker">ENGINEERING &bull; UNDER THE HOOD</div>
            <div style="font-size: 22px; font-weight: 800; color: #FFF; margin-bottom: 16px;">
                Classical Image and Video Processing Core
            </div>
            <div class="feat-explain-grid">
                <div class="feat-explain-card">
                    <div class="feat-explain-title">&#9672; Spatial Sharpening</div>
                    <div class="feat-explain-desc">
                        Employs a 2D discrete Laplacian convolution kernel boosting high spatial frequencies along image contours.
                    </div>
                </div>
                <div class="feat-explain-card">
                    <div class="feat-explain-title">&#9672; Dynamic Range Contrast</div>
                    <div class="feat-explain-desc">
                        Centered linear transformation scaling pixel deviations from intensity midpoint 128 with saturation clipping.
                    </div>
                </div>
                <div class="feat-explain-card">
                    <div class="feat-explain-title">&#9672; HSV Chrominance</div>
                    <div class="feat-explain-desc">
                        Color space conversion isolating pure Hue from Saturation and Value to boost vividness without hue drift.
                    </div>
                </div>
                <div class="feat-explain-card">
                    <div class="feat-explain-title">&#9672; Gaussian Synthesis</div>
                    <div class="feat-explain-desc">
                        Controlled zero-mean Box-Muller Gaussian noise simulation creating authentic cinematic film grain.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 7. HOW IT WORKS SECTION (Nav Anchor: #how-it-works)
    # --------------------------------------------------------------------------
    st.markdown("<div id='how-it-works'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-top: 36px; padding: 24px; background: rgba(14, 21, 38, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px;">
            <div class="ws-title-kicker">WORKFLOW</div>
            <div style="font-size: 18px; font-weight: 800; color: #FFF; margin-bottom: 12px;">
                Four-Step Interactive Calibration
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; font-size: 12px; color: #8A9BB8;">
                <div><b style="color: #FF6B35;">01. Load Source</b><br>Initializes comic hero sample or user uploaded frame in memory.</div>
                <div><b style="color: #FF6B35;">02. Live Filtering</b><br>Applies 5 real-time OpenCV transforms with mathematical feedback.</div>
                <div><b style="color: #FF6B35;">03. Split Inspection</b><br>Drag the 60fps divider lens to verify edge details and color balance.</div>
                <div><b style="color: #FF6B35;">04. Export Frame</b><br>Download full-resolution lossless PNG output directly to disk.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 8. PRICING & COURSEWARE NOTIFICATION (Nav Anchor: #pricing)
    # --------------------------------------------------------------------------
    st.markdown("<div id='pricing'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-top: 30px; text-align: center; padding: 24px; background: rgba(255, 107, 53, 0.06); border: 1px solid rgba(255, 107, 53, 0.2); border-radius: 14px;">
            <div style="font-size: 14px; font-weight: 800; color: #FFB703; margin-bottom: 4px;">
                ACADEMIC &amp; OPEN SOURCE COURSEWARE &bull; 100% FREE
            </div>
            <div style="font-size: 12px; color: #94A3B8;">
                Built for Image and Video Processing (IVP) mini-project coursework. Runs locally with zero cloud dependencies or external APIs.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 9. FOOTER
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <footer class="footer-wrap">
            <div class="footer-brand">
                REVEAL STUDIO // NETRAYA IMAGE STUDIO &bull; IVP COURSEWARE
            </div>
            <div>Python &bull; OpenCV &bull; NumPy &bull; Streamlit &bull; Lossless Image Processing</div>
        </footer>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
